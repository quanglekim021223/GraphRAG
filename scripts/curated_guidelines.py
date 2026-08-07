"""Admin CLI for discovering, ingesting and reviewing curated guidelines."""
import argparse
import json
import sys
from typing import Any, Dict

from src.config.settings import Config
from src.handlers.curated_guidelines import (
    CuratedGuidelineStore,
    DocumentMetadata,
    EmbeddingProviderError,
    GuidelineIngestionError,
    OpenAIEmbedder,
    ingest_guideline,
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage the reviewed medical-guideline corpus"
    )
    parser.add_argument("--database", help="Override CURATED_GUIDELINE_DB_PATH")
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser(
        "discover", help="Use Tavily only to identify review candidates"
    )
    discover.add_argument("question")

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
    database_path = args.database or config.curated_guideline_db_path
    store = CuratedGuidelineStore(database_path)
    try:
        if args.command == "discover":
            result = search_medical_guidelines(
                question=args.question,
                api_key=config.tavily_api_key,
                max_results=config.medical_search_max_results,
                min_score=config.medical_search_min_score,
                actor_id="curated-ingestion-admin",
                cache_ttl_seconds=config.medical_search_cache_ttl_seconds,
                rate_limit_per_minute=config.medical_search_rate_limit_per_minute,
                daily_budget=config.medical_search_daily_budget,
                max_retries=config.medical_search_max_retries,
                max_retry_delay_seconds=config.medical_search_max_retry_delay_seconds,
                circuit_failure_threshold=(
                    config.medical_search_circuit_failure_threshold
                ),
                circuit_cooldown_seconds=(
                    config.medical_search_circuit_cooldown_seconds
                ),
            )
            candidates: Dict[str, Any] = {
                "status": result["status"],
                "notice": "Discovery only; no result is approved automatically.",
                "candidates": result.get("evidence", []),
            }
            _print(candidates)
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
    except (EmbeddingProviderError, GuidelineIngestionError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
