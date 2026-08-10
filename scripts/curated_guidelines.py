"""Admin CLI for discovering, ingesting and reviewing curated guidelines."""
import argparse
import json
import sys
from typing import Any, Dict

from psycopg import Error as PostgresError
from psycopg_pool import PoolTimeout

from src.config.settings import Config
from src.handlers.curated_guidelines import (
    DEFAULT_PREWARM_TOPICS,
    CuratedGuidelineStore,
    DocumentMetadata,
    EmbeddingProviderError,
    GuidelineIngestionError,
    OpenAIEmbedder,
    ingest_guideline,
    prewarm_guideline_corpus,
)
from src.handlers.medical_guideline_search import search_medical_guidelines


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _metadata(args: argparse.Namespace) -> DocumentMetadata:
    return DocumentMetadata(
        title=args.title,
        publisher=args.publisher,
        publication_date=args.publication_date,
        version=args.version,
        effective_from=args.effective_from,
        effective_until=args.effective_until,
    )


def _search_options(config: Config, actor_id: str) -> Dict[str, Any]:
    return {
        "min_score": config.medical_search_min_score,
        "actor_id": actor_id,
        "cache_ttl_seconds": config.medical_search_cache_ttl_seconds,
        "rate_limit_per_minute": config.medical_search_rate_limit_per_minute,
        "daily_budget": config.medical_search_daily_budget,
        "max_retries": config.medical_search_max_retries,
        "max_retry_delay_seconds": config.medical_search_max_retry_delay_seconds,
        "circuit_failure_threshold": (
            config.medical_search_circuit_failure_threshold
        ),
        "circuit_cooldown_seconds": (
            config.medical_search_circuit_cooldown_seconds
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage the reviewed medical-guideline corpus"
    )
    parser.add_argument(
        "--postgres-uri",
        help="Override POSTGRES_URI for this admin command",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser(
        "discover", help="Use Tavily only to identify review candidates"
    )
    discover.add_argument("question")

    prewarm = subparsers.add_parser(
        "prewarm",
        help="Populate missing common topics through the trusted runtime pipeline",
    )
    prewarm.add_argument(
        "--topic",
        dest="topics",
        action="append",
        help="Custom de-identified topic; repeat for multiple topics",
    )

    ingest = subparsers.add_parser(
        "ingest", help="Download and hash a candidate as pending_review"
    )
    ingest.add_argument("--url", required=True)
    ingest.add_argument("--title", required=True)
    ingest.add_argument("--publisher", required=True)
    ingest.add_argument("--publication-date", required=True)
    ingest.add_argument("--version", required=True)
    ingest.add_argument("--effective-from", required=True)
    ingest.add_argument("--effective-until")

    list_parser = subparsers.add_parser("list", help="List corpus documents")
    list_parser.add_argument("--status")

    show = subparsers.add_parser("show", help="Show metadata and review hash")
    show.add_argument("document_id")

    approve = subparsers.add_parser(
        "approve", help="Extract, embed and activate an inspected hash"
    )
    approve.add_argument("document_id")
    approve.add_argument("--reviewer", required=True)
    approve.add_argument("--expected-hash", required=True)

    reject = subparsers.add_parser("reject", help="Reject a pending candidate")
    reject.add_argument("document_id")
    reject.add_argument("--reviewer", required=True)

    withdraw = subparsers.add_parser("withdraw", help="Withdraw an active document")
    withdraw.add_argument("document_id")
    withdraw.add_argument("--reviewer", required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    config = Config()
    postgres_uri = args.postgres_uri or config.postgres_uri
    store = None
    try:
        if args.command == "discover":
            result = search_medical_guidelines(
                question=args.question,
                api_key=config.tavily_api_key,
                max_results=config.medical_search_max_results,
                **_search_options(config, "curated-ingestion-admin"),
            )
            candidates: Dict[str, Any] = {
                "status": result["status"],
                "notice": "Discovery only; no result is approved automatically.",
                "candidates": result.get("evidence", []),
            }
            _print(candidates)
        else:
            store = CuratedGuidelineStore(postgres_uri)
        if args.command == "prewarm":
            embedder = OpenAIEmbedder(
                config.curated_embedding_endpoint,
                config.github_token,
                config.curated_embedding_model,
            )
            _print(prewarm_guideline_corpus(
                topics=args.topics or DEFAULT_PREWARM_TOPICS,
                retrieval_options={
                    "postgres_uri": postgres_uri,
                    "endpoint": config.curated_embedding_endpoint,
                    "api_key": config.github_token,
                    "embedding_model": config.curated_embedding_model,
                    "tavily_api_key": config.tavily_api_key,
                    "top_k": config.curated_retrieval_top_k,
                    "min_score": config.curated_retrieval_min_score,
                    "discovery_max_results": config.curated_discovery_max_results,
                    "auto_ingest_max_documents": (
                        config.curated_auto_ingest_max_documents
                    ),
                    "search_options": _search_options(
                        config, "curated-prewarm-admin"
                    ),
                    "embedder": embedder,
                    "store": store,
                },
            ))
        elif args.command == "ingest":
            _print(ingest_guideline(
                store,
                args.url,
                _metadata(args),
                config.curated_embedding_model,
            ))
        elif args.command == "list":
            _print(store.list_documents(args.status))
        elif args.command == "show":
            _print(store.get_review_bundle(args.document_id))
        elif args.command == "approve":
            embedder = OpenAIEmbedder(
                config.curated_embedding_endpoint,
                config.github_token,
                config.curated_embedding_model,
            )
            _print(store.approve(
                args.document_id,
                args.reviewer,
                args.expected_hash,
                embedder,
            ))
        elif args.command == "reject":
            _print(store.reject(args.document_id, args.reviewer))
        elif args.command == "withdraw":
            _print(store.withdraw(args.document_id, args.reviewer))
    except (
        EmbeddingProviderError,
        GuidelineIngestionError,
        PoolTimeout,
        PostgresError,
        ValueError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    finally:
        if store is not None:
            store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
