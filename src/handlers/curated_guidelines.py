"""Trusted medical-guideline ingestion and section-level retrieval.

The answer path reads either internally approved documents or documents
automatically admitted by the exact official-source policy. On a corpus miss,
the bounded fallback can discover, download, validate, chunk and embed an
official document before retrying retrieval. Retrieval remains extractive and
deterministic: the model never rewrites a guideline claim.
"""
import atexit
import hashlib
import hmac
import ipaddress
import math
import re
import socket
from dataclasses import dataclass
from datetime import date, datetime, timezone
from functools import lru_cache
from html.parser import HTMLParser
from io import BytesIO
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from psycopg import Error as PostgresError
from psycopg import IntegrityError
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, PoolTimeout

from src.handlers.medical_guideline_search import (
    SENSITIVE_SEARCH_REFUSAL,
    contains_sensitive_patient_data,
    is_approved_source_url,
    resolve_approved_final_url,
    search_medical_guidelines,
)
from src.handlers.security_guardrails import audit_event
from src.helpers.logging_config import logger


MAX_DOCUMENT_BYTES = 10_000_000
MAX_SECTIONS_PER_DOCUMENT = 1000
TRUSTED_OFFICIAL_STATUS = "trusted_official"
INTERNAL_APPROVED_STATUS = "approved"
TRUSTED_OFFICIAL_ACTOR = "system:trusted-official-policy-v1"
RETRIEVABLE_REVIEW_STATUSES = (
    INTERNAL_APPROVED_STATUS,
    TRUSTED_OFFICIAL_STATUS,
)
SUPPORTED_CONTENT_TYPES = {
    "application/pdf",
    "text/html",
    "application/xhtml+xml",
    "text/plain",
}
CURATED_CORPUS_EMPTY = (
    "Kho hướng dẫn y khoa chưa có tài liệu nội bộ đã duyệt hoặc tài liệu chính "
    "thức đủ điều kiện tự động nạp."
)
CURATED_RETRIEVAL_UNAVAILABLE = (
    "Kho hướng dẫn y khoa hiện không thể truy xuất. Hệ thống sẽ không tạo "
    "câu trả lời không có nguồn."
)
CURATED_NO_RESULT = (
    "Không tìm thấy section đủ liên quan trong các hướng dẫn đang được phép "
    "sử dụng. Vui lòng hỏi cụ thể hơn hoặc kiểm tra tài liệu nguồn."
)
SOURCE_DIFFERENCE_WARNING = (
    "Các section được hiển thị riêng theo thứ tự relevance; hệ thống không tự "
    "giải quyết khác biệt giữa các guideline."
)


class GuidelineIngestionError(ValueError):
    """Raised when a candidate cannot safely enter the trusted corpus."""


class EmbeddingProviderError(RuntimeError):
    """Raised when embeddings are unavailable or incomplete."""


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        del req, fp, code, msg, headers, newurl
        return None


_DOWNLOAD_OPENER = build_opener(_NoRedirectHandler())


@dataclass(frozen=True)
class DownloadedDocument:
    """Bounded content downloaded from a validated final URL."""

    source_url: str
    final_url: str
    content_type: str
    content: bytes


@dataclass(frozen=True)
class DocumentMetadata:
    """Version metadata supplied by an admin or strict official-source policy."""

    title: str
    publisher: str
    publication_date: str
    version: str
    effective_from: str
    effective_until: Optional[str] = None


class OpenAIEmbedder:
    """Minimal batched embedding adapter using the repository's OpenAI client."""

    def __init__(self, endpoint: str, api_key: str, model: str):
        if not api_key:
            raise ValueError("Embedding API key is not configured")
        try:
            from openai import OpenAI  # pylint: disable=import-outside-toplevel
        except ImportError as error:
            raise EmbeddingProviderError("OpenAI dependency is unavailable") from error
        self.model = model
        self.client = OpenAI(base_url=endpoint, api_key=api_key)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors: List[List[float]] = []
        for start in range(0, len(texts), 64):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=list(texts[start:start + 64]),
                )
            except Exception as error:  # provider SDK exposes versioned exceptions
                raise EmbeddingProviderError("Embedding request failed") from error
            vectors.extend(
                list(item.embedding)
                for item in sorted(response.data, key=lambda item: item.index)
            )
        if len(vectors) != len(texts):
            raise EmbeddingProviderError(
                "Embedding provider returned an incomplete batch"
            )
        return vectors


class CuratedGuidelineStore:
    """PostgreSQL catalog containing immutable versions and embeddings.

    Embeddings remain ordinary PostgreSQL arrays and are ranked in Python. This
    keeps the current small-corpus behavior while removing the single-host
    SQLite dependency; pgvector can be added later without changing callers.
    """

    def __init__(
        self,
        postgres_uri: str,
        pool: Optional[ConnectionPool] = None,
    ) -> None:
        if not postgres_uri and pool is None:
            raise ValueError("PostgreSQL URI is required")
        self._owns_pool = pool is None
        self.pool = pool or ConnectionPool(
            conninfo=postgres_uri,
            min_size=1,
            max_size=5,
            timeout=10,
            kwargs={
                "autocommit": False,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
            open=True,
        )
        if self._owns_pool:
            self.pool.wait(timeout=10)
        self._initialize()
        if self._owns_pool:
            atexit.register(self.close)

    def close(self) -> None:
        """Close an owned pool; injected pools remain owned by their caller."""
        if self._owns_pool:
            self.pool.close()

    def _initialize(self) -> None:
        statements = (
            """
                CREATE TABLE IF NOT EXISTS guideline_documents (
                    document_id TEXT PRIMARY KEY,
                    source_url TEXT NOT NULL,
                    final_url TEXT NOT NULL,
                    title TEXT NOT NULL,
                    publisher TEXT NOT NULL,
                    publication_date DATE NOT NULL,
                    version TEXT NOT NULL,
                    effective_from DATE NOT NULL,
                    effective_until DATE,
                    review_status TEXT NOT NULL,
                    effective_status TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    raw_content BYTEA NOT NULL,
                    embedding_model TEXT NOT NULL,
                    downloaded_at TIMESTAMPTZ NOT NULL,
                    reviewed_at TIMESTAMPTZ,
                    reviewed_by TEXT,
                    UNIQUE(source_url, version)
                )
            """,
            """
                CREATE TABLE IF NOT EXISTS guideline_sections (
                    section_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    heading TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    embedding DOUBLE PRECISION[] NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES guideline_documents(document_id)
                        ON DELETE CASCADE,
                    UNIQUE(document_id, ordinal)
                )
            """,
            """
                CREATE INDEX IF NOT EXISTS idx_guideline_effective
                ON guideline_documents(
                    review_status, effective_status, effective_from, effective_until
                )
            """,
        )
        with self.pool.connection() as connection:
            for statement in statements:
                connection.execute(statement)

    def add_pending_document(
        self,
        downloaded: DownloadedDocument,
        metadata: DocumentMetadata,
        embedding_model: str,
        downloaded_at: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Persist one immutable raw version without adding it to the index."""
        _validate_metadata(metadata)
        if not embedding_model.strip():
            raise GuidelineIngestionError("Embedding model is required")

        content_hash = hashlib.sha256(downloaded.content).hexdigest()
        identity = f"{downloaded.source_url}|{metadata.version}|{content_hash}"
        document_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
        timestamp = (downloaded_at or datetime.now(timezone.utc)).astimezone(
            timezone.utc
        )

        try:
            with self.pool.connection() as connection:
                connection.execute(
                    """
                    INSERT INTO guideline_documents (
                        document_id, source_url, final_url, title, publisher,
                        publication_date, version, effective_from, effective_until,
                        review_status, effective_status, content_type, content_hash,
                        raw_content, embedding_model, downloaded_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """,
                    (
                        document_id,
                        downloaded.source_url,
                        downloaded.final_url,
                        metadata.title.strip(),
                        metadata.publisher.strip(),
                        metadata.publication_date,
                        metadata.version.strip(),
                        metadata.effective_from,
                        metadata.effective_until,
                        "pending_review",
                        "pending",
                        downloaded.content_type,
                        content_hash,
                        downloaded.content,
                        embedding_model,
                        timestamp,
                    ),
                )
        except IntegrityError as error:
            raise GuidelineIngestionError(
                "This source URL and version already exist and are immutable"
            ) from error

        audit_event(
            "curated_guideline_ingested",
            document_id=document_id,
            content_hash=content_hash,
        )
        return self.get_document(document_id)

    def approve(
        self,
        document_id: str,
        reviewer: str,
        expected_content_hash: str,
        embedder: Any,
        reviewed_at: Optional[datetime] = None,
        max_chunk_chars: int = 2200,
        chunk_overlap_chars: int = 200,
    ) -> Dict[str, Any]:
        """Activate one hash after explicit internal review."""
        if not reviewer.strip():
            raise GuidelineIngestionError("Reviewer identity is required")
        return self._activate_document(
            document_id=document_id,
            actor=reviewer.strip(),
            expected_content_hash=expected_content_hash,
            embedder=embedder,
            review_status=INTERNAL_APPROVED_STATUS,
            reviewed_at=reviewed_at,
            max_chunk_chars=max_chunk_chars,
            chunk_overlap_chars=chunk_overlap_chars,
        )

    def activate_trusted_official(
        self,
        document_id: str,
        expected_content_hash: str,
        embedder: Any,
        activated_at: Optional[datetime] = None,
        max_chunk_chars: int = 2200,
        chunk_overlap_chars: int = 200,
    ) -> Dict[str, Any]:
        """Auto-activate only a document whose stored URLs remain allowlisted."""
        with self.pool.connection() as connection:
            row = connection.execute(
                """
                SELECT source_url, final_url FROM guideline_documents
                WHERE document_id = %s
                """,
                (document_id,),
            ).fetchone()
        if row is None:
            raise GuidelineIngestionError("Document does not exist")
        if not (
            is_approved_source_url(row["source_url"])
            and is_approved_source_url(row["final_url"])
        ):
            raise GuidelineIngestionError(
                "Trusted-official activation requires exact allowlisted URLs"
            )
        return self._activate_document(
            document_id=document_id,
            actor=TRUSTED_OFFICIAL_ACTOR,
            expected_content_hash=expected_content_hash,
            embedder=embedder,
            review_status=TRUSTED_OFFICIAL_STATUS,
            reviewed_at=activated_at,
            max_chunk_chars=max_chunk_chars,
            chunk_overlap_chars=chunk_overlap_chars,
        )

    def _activate_document(
        self,
        document_id: str,
        actor: str,
        expected_content_hash: str,
        embedder: Any,
        review_status: str,
        reviewed_at: Optional[datetime],
        max_chunk_chars: int,
        chunk_overlap_chars: int,
    ) -> Dict[str, Any]:
        """Extract, embed and activate one immutable pending document."""
        if review_status not in RETRIEVABLE_REVIEW_STATUSES:
            raise GuidelineIngestionError("Unsupported trusted review status")
        with self.pool.connection() as connection:
            row = connection.execute(
                "SELECT * FROM guideline_documents WHERE document_id = %s",
                (document_id,),
            ).fetchone()
        if row is None:
            raise GuidelineIngestionError("Document does not exist")
        if row["review_status"] != "pending_review":
            raise GuidelineIngestionError("Only pending documents can be approved")
        if not _constant_time_equal(row["content_hash"], expected_content_hash):
            raise GuidelineIngestionError(
                "Reviewed content hash does not match the downloaded document"
            )
        if row["embedding_model"] != embedder.model:
            raise GuidelineIngestionError(
                "Activation embedder does not match the candidate embedding model"
            )

        extracted = extract_sections(bytes(row["raw_content"]), row["content_type"])
        sections = chunk_sections(
            extracted, max_chunk_chars, chunk_overlap_chars
        )
        if not sections:
            raise GuidelineIngestionError("No usable full text was extracted")
        if len(sections) > MAX_SECTIONS_PER_DOCUMENT:
            raise GuidelineIngestionError(
                "Document exceeds the section indexing limit"
            )
        try:
            embeddings = embedder.embed(
                [f"{heading}\n{content}" for heading, content in sections]
            )
        except EmbeddingProviderError as error:
            raise GuidelineIngestionError(
                "Trusted document could not be embedded"
            ) from error
        _validate_embedding_batch(sections, embeddings)
        timestamp = (reviewed_at or datetime.now(timezone.utc)).astimezone(
            timezone.utc
        )

        with self.pool.connection() as connection:
            current = connection.execute(
                """
                SELECT review_status, content_hash FROM guideline_documents
                WHERE document_id = %s
                FOR UPDATE
                """,
                (document_id,),
            ).fetchone()
            if (
                current is None
                or current["review_status"] != "pending_review"
                or not _constant_time_equal(
                    current["content_hash"], expected_content_hash
                )
            ):
                raise GuidelineIngestionError(
                    "Candidate changed state while approval was in progress"
                )
            for ordinal, ((heading, content), vector) in enumerate(
                zip(sections, embeddings), start=1
            ):
                section_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                section_id = hashlib.sha256(
                    f"{document_id}|{ordinal}|{section_hash}".encode("utf-8")
                ).hexdigest()[:32]
                connection.execute(
                    """
                    INSERT INTO guideline_sections (
                        section_id, document_id, ordinal, heading, content,
                        content_hash, embedding
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        section_id,
                        document_id,
                        ordinal,
                        heading,
                        content,
                        section_hash,
                        list(vector),
                    ),
                )
            connection.execute(
                """
                UPDATE guideline_documents
                SET effective_status = 'superseded'
                WHERE source_url = %s AND document_id <> %s
                  AND review_status IN ('approved', 'trusted_official')
                  AND effective_status = 'active'
                """,
                (row["source_url"], document_id),
            )
            connection.execute(
                """
                UPDATE guideline_documents
                SET review_status = %s, effective_status = 'active',
                    reviewed_at = %s, reviewed_by = %s
                WHERE document_id = %s
                """,
                (review_status, timestamp, actor, document_id),
            )
        audit_event(
            "curated_guideline_activated",
            document_id=document_id,
            review_status=review_status,
            actor_hash=hashlib.sha256(actor.encode("utf-8")).hexdigest(),
            section_count=len(sections),
        )
        return self.get_document(document_id)

    def reject(self, document_id: str, reviewer: str) -> Dict[str, Any]:
        return self._change_review_state(document_id, reviewer, "rejected", "rejected")

    def withdraw(self, document_id: str, reviewer: str) -> Dict[str, Any]:
        return self._change_review_state(document_id, reviewer, "approved", "withdrawn")

    def _change_review_state(
        self,
        document_id: str,
        reviewer: str,
        review_status: str,
        effective_status: str,
    ) -> Dict[str, Any]:
        if not reviewer.strip():
            raise GuidelineIngestionError("Reviewer identity is required")
        with self.pool.connection() as connection:
            row = connection.execute(
                """
                SELECT review_status FROM guideline_documents
                WHERE document_id = %s FOR UPDATE
                """,
                (document_id,),
            ).fetchone()
            if row is None:
                raise GuidelineIngestionError("Document does not exist")
            if (
                effective_status == "withdrawn"
                and row["review_status"] not in RETRIEVABLE_REVIEW_STATUSES
            ):
                raise GuidelineIngestionError(
                    "Only active trusted documents can be withdrawn"
                )
            if effective_status == "rejected" and row["review_status"] != "pending_review":
                raise GuidelineIngestionError("Only pending documents can be rejected")
            if effective_status == "withdrawn":
                review_status = row["review_status"]
            connection.execute(
                """
                UPDATE guideline_documents
                SET review_status = %s, effective_status = %s, reviewed_at = %s,
                    reviewed_by = %s WHERE document_id = %s
                """,
                (
                    review_status,
                    effective_status,
                    datetime.now(timezone.utc),
                    reviewer.strip(),
                    document_id,
                ),
            )
        audit_event(
            f"curated_guideline_{effective_status}", document_id=document_id
        )
        return self.get_document(document_id)

    def get_document(self, document_id: str) -> Dict[str, Any]:
        with self.pool.connection() as connection:
            row = connection.execute(
                """
                SELECT d.*, COUNT(s.section_id) AS section_count
                FROM guideline_documents d
                LEFT JOIN guideline_sections s ON s.document_id = d.document_id
                WHERE d.document_id = %s GROUP BY d.document_id
                """,
                (document_id,),
            ).fetchone()
        if row is None:
            raise GuidelineIngestionError("Document does not exist")
        result = dict(row)
        result.pop("raw_content", None)
        return result

    def get_review_bundle(self, document_id: str) -> Dict[str, Any]:
        """Return immutable metadata plus extracted section text for review."""
        document = self.get_document(document_id)
        with self.pool.connection() as connection:
            sections = [
                dict(row) for row in connection.execute(
                    """
                    SELECT section_id, ordinal, heading, content, content_hash
                    FROM guideline_sections WHERE document_id = %s
                    ORDER BY ordinal
                    """,
                    (document_id,),
                )
            ]
            if not sections:
                raw = connection.execute(
                    """
                    SELECT raw_content, content_type FROM guideline_documents
                    WHERE document_id = %s
                    """,
                    (document_id,),
                ).fetchone()
                preview = chunk_sections(
                    extract_sections(bytes(raw["raw_content"]), raw["content_type"])
                )
                sections = [
                    {
                        "section_id": None,
                        "ordinal": ordinal,
                        "heading": heading,
                        "content": content,
                        "content_hash": hashlib.sha256(
                            content.encode("utf-8")
                        ).hexdigest(),
                    }
                    for ordinal, (heading, content) in enumerate(preview, start=1)
                ]
        return {"document": document, "sections": sections}

    def list_documents(self, review_status: Optional[str] = None) -> List[Dict[str, Any]]:
        query = """
            SELECT document_id, source_url, final_url, title, publisher,
                   publication_date, version, effective_from, effective_until,
                   review_status, effective_status, content_hash, embedding_model,
                   downloaded_at, reviewed_at, reviewed_by
            FROM guideline_documents
        """
        parameters: Tuple[Any, ...] = ()
        if review_status:
            query += " WHERE review_status = %s"
            parameters = (review_status,)
        query += " ORDER BY downloaded_at DESC"
        with self.pool.connection() as connection:
            return [dict(row) for row in connection.execute(query, parameters)]

    def has_effective_documents(
        self, embedding_model: str, on_date: Optional[date] = None
    ) -> bool:
        today = on_date or date.today()
        with self.pool.connection() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM guideline_documents
                WHERE review_status IN ('approved', 'trusted_official')
                  AND effective_status = 'active'
                  AND embedding_model = %s AND effective_from <= %s
                  AND (effective_until IS NULL OR effective_until >= %s)
                LIMIT 1
                """,
                (embedding_model, today, today),
            ).fetchone()
        return row is not None

    def search(
        self,
        query_vector: Sequence[float],
        embedding_model: str,
        top_k: int,
        min_score: float,
        on_date: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """Rank effective sections by cosine similarity in the small corpus."""
        today = on_date or date.today()
        with self.pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT s.section_id, s.ordinal, s.heading, s.content,
                       s.content_hash AS section_hash, s.embedding,
                       d.document_id, d.title, d.publisher, d.publication_date,
                       d.version, d.effective_from, d.effective_until,
                       d.content_hash AS document_hash, d.final_url,
                       d.review_status, d.reviewed_at, d.reviewed_by
                FROM guideline_sections s
                JOIN guideline_documents d ON d.document_id = s.document_id
                WHERE d.review_status IN ('approved', 'trusted_official')
                  AND d.effective_status = 'active'
                  AND d.embedding_model = %s AND d.effective_from <= %s
                  AND (d.effective_until IS NULL OR d.effective_until >= %s)
                """,
                (embedding_model, today, today),
            ).fetchall()

        ranked = []
        for row in rows:
            vector = row["embedding"]
            score = _cosine_similarity(query_vector, vector)
            if score < min_score:
                continue
            item = dict(row)
            item.pop("embedding")
            item["score"] = score
            ranked.append(item)
        ranked.sort(key=lambda item: (-item["score"], item["section_id"]))
        return ranked[:max(1, min(int(top_k), 10))]


@lru_cache(maxsize=4)
def get_curated_guideline_store(postgres_uri: str) -> CuratedGuidelineStore:
    """Reuse one bounded PostgreSQL pool per configured corpus database."""
    return CuratedGuidelineStore(postgres_uri)


def download_approved_document(
    source_url: str,
    head_opener: Optional[Callable[..., Any]] = None,
    get_opener: Optional[Callable[..., Any]] = None,
    resolver: Optional[Callable[..., Any]] = None,
    max_bytes: int = MAX_DOCUMENT_BYTES,
) -> DownloadedDocument:
    """Resolve an allowlisted URL, then download a bounded approved content type."""
    final_url = resolve_approved_final_url(
        source_url,
        opener=head_opener,
        resolver=resolver,
    )
    if not final_url:
        raise GuidelineIngestionError("Source URL or redirect chain is not approved")
    if not _host_has_only_public_addresses(
        urlparse(final_url).hostname or "", resolver or socket.getaddrinfo
    ):
        raise GuidelineIngestionError("Final source host does not resolve publicly")

    request = Request(
        final_url,
        headers={
            "Accept": "application/pdf,text/html,text/plain",
            "User-Agent": "HealthcareGraphRAG-Ingestion/1.0",
        },
        method="GET",
    )
    open_document = get_opener or _DOWNLOAD_OPENER.open
    try:
        with open_document(request, timeout=20) as response:
            status = int(getattr(response, "status", 200))
            response_url = str(
                response.geturl() if hasattr(response, "geturl") else final_url
            )
            if not 200 <= status < 300 or response_url != final_url:
                raise GuidelineIngestionError(
                    "GET response changed URL or returned a non-success status"
                )
            content_type = str(
                getattr(response, "headers", {}).get("Content-Type", "")
            ).split(";", 1)[0].strip().lower()
            if content_type not in SUPPORTED_CONTENT_TYPES:
                raise GuidelineIngestionError(
                    f"Unsupported document content type: {content_type or 'missing'}"
                )
            content = response.read(max_bytes + 1)
    except GuidelineIngestionError:
        raise
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
        raise GuidelineIngestionError(
            f"Controlled document download failed: {type(error).__name__}"
        ) from error
    if not content or len(content) > max_bytes:
        raise GuidelineIngestionError("Document is empty or exceeds the size limit")
    return DownloadedDocument(source_url, final_url, content_type, content)


def ingest_guideline(
    store: CuratedGuidelineStore,
    source_url: str,
    metadata: DocumentMetadata,
    embedding_model: str,
    downloader: Callable[..., DownloadedDocument] = download_approved_document,
    **download_kwargs: Any,
) -> Dict[str, Any]:
    """Download and hash a candidate; extraction/indexing waits for approval."""
    downloaded = downloader(source_url, **download_kwargs)
    return store.add_pending_document(
        downloaded=downloaded,
        metadata=metadata,
        embedding_model=embedding_model,
    )


def auto_ingest_trusted_guidelines(
    question: str,
    store: CuratedGuidelineStore,
    embedder: Any,
    tavily_api_key: str,
    discovery_max_results: int = 5,
    max_documents: int = 3,
    search_options: Optional[Dict[str, Any]] = None,
    searcher: Callable[..., Dict[str, Any]] = search_medical_guidelines,
    downloader: Callable[[str], DownloadedDocument] = download_approved_document,
    on_date: Optional[date] = None,
) -> Dict[str, Any]:
    """Discover and auto-index strict official-source documents on a miss.

    The policy never accepts a provider snippet as corpus content. It requires
    an exact allowlisted final URL and an ISO publication date, downloads the
    full document through the controlled downloader, versions it by full-content
    hash, then activates it as ``trusted_official``.
    """
    if contains_sensitive_patient_data(question):
        return {"status": "privacy_denied", "ingested": [], "skipped": 0}

    safe_discovery_max = max(1, min(int(discovery_max_results), 5))
    safe_max_documents = min(
        max(1, min(int(max_documents), 3)), safe_discovery_max
    )
    options = dict(search_options or {})
    options["max_results"] = safe_discovery_max
    discovery = searcher(
        question=question,
        api_key=tavily_api_key,
        **options,
    )
    if discovery.get("status") != "success":
        return {
            "status": discovery.get("status", "unavailable"),
            "ingested": [],
            "skipped": 0,
        }

    today = on_date or date.today()
    ingested: List[Dict[str, Any]] = []
    skipped = 0
    discovered = discovery.get("evidence", [])
    if not isinstance(discovered, list):
        discovered = []
    candidates = sorted(
        discovered[:safe_discovery_max], key=_trusted_candidate_rank
    )
    for candidate in candidates:
        if len(ingested) >= safe_max_documents:
            break
        if not isinstance(candidate, dict):
            skipped += 1
            continue
        final_url = str(candidate.get("final_url") or candidate.get("url") or "")
        title = str(candidate.get("title") or "").strip()
        publisher = str(candidate.get("source_name") or "").strip()
        publication_date = _strict_publication_date(
            candidate.get("publication_date"), today
        )
        if not (
            final_url
            and title
            and publisher
            and publication_date
            and is_approved_source_url(final_url)
        ):
            skipped += 1
            audit_event(
                "trusted_guideline_candidate_skipped",
                reason="missing_or_untrusted_metadata",
            )
            continue

        try:
            downloaded = downloader(final_url)
            content_hash = hashlib.sha256(downloaded.content).hexdigest()
            version = f"official-{publication_date}-{content_hash[:12]}"
            document = store.add_pending_document(
                downloaded=downloaded,
                metadata=DocumentMetadata(
                    title=title,
                    publisher=publisher,
                    publication_date=publication_date,
                    version=version,
                    effective_from=publication_date,
                ),
                embedding_model=embedder.model,
            )
            activated = store.activate_trusted_official(
                document_id=document["document_id"],
                expected_content_hash=document["content_hash"],
                embedder=embedder,
            )
            ingested.append({
                "document_id": activated["document_id"],
                "final_url": activated["final_url"],
                "version": activated["version"],
                "content_hash": activated["content_hash"],
                "review_status": activated["review_status"],
            })
        except (GuidelineIngestionError, OSError, ValueError) as error:
            skipped += 1
            audit_event(
                "trusted_guideline_candidate_skipped",
                reason=type(error).__name__,
            )

    status = "success" if ingested else "not_found"
    audit_event(
        "trusted_guideline_auto_ingest_completed",
        status=status,
        discovered_count=len(candidates),
        ingested_count=len(ingested),
        skipped_count=skipped,
    )
    return {
        "status": status,
        "discovered": len(candidates),
        "ingested": ingested,
        "skipped": skipped,
    }


def retrieve_curated_guidelines(
    question: str,
    postgres_uri: str,
    endpoint: str,
    api_key: str,
    embedding_model: str,
    top_k: int = 3,
    min_score: float = 0.45,
    embedder: Optional[Any] = None,
    on_date: Optional[date] = None,
    store: Optional[CuratedGuidelineStore] = None,
) -> Dict[str, Any]:
    """Return only effective section text with versioned, section-level citations."""
    if contains_sensitive_patient_data(question):
        audit_event("curated_guideline_rejected_sensitive_input", level="warning")
        return {"status": "privacy_denied", "response": SENSITIVE_SEARCH_REFUSAL}
    try:
        active_store = store or get_curated_guideline_store(postgres_uri)
        active_embedder = embedder or OpenAIEmbedder(endpoint, api_key, embedding_model)
        model = active_embedder.model
        if not active_store.has_effective_documents(model, on_date):
            audit_event("curated_guideline_corpus_empty")
            return {"status": "corpus_empty", "response": CURATED_CORPUS_EMPTY}
        query_vector = active_embedder.embed([question.strip()])[0]
        safe_top_k = max(1, min(int(top_k), 3))
        matches = active_store.search(
            query_vector=query_vector,
            embedding_model=model,
            top_k=safe_top_k,
            min_score=min_score,
            on_date=on_date,
        )
    except (
        EmbeddingProviderError,
        GuidelineIngestionError,
        OSError,
        PostgresError,
        PoolTimeout,
        ValueError,
    ) as error:
        logger.warning("Curated guideline retrieval failed: %s", type(error).__name__)
        audit_event(
            "curated_guideline_retrieval_unavailable",
            reason=type(error).__name__,
        )
        return {"status": "unavailable", "response": CURATED_RETRIEVAL_UNAVAILABLE}

    if not matches:
        audit_event("curated_guideline_no_relevant_section")
        return {"status": "not_found", "response": CURATED_NO_RESULT}

    retrieved_at = datetime.now(timezone.utc).isoformat()
    lines = ["Nội dung từ các guideline đang được phép sử dụng:"]
    evidence = []
    for index, match in enumerate(matches, start=1):
        evidence_id = f"G{index}"
        trust_label = (
            "Nội bộ đã duyệt"
            if match["review_status"] == INTERNAL_APPROVED_STATUS
            else "Nguồn chính thức tự động xác minh"
        )
        lines.extend([
            "",
            f'[{evidence_id}] {match["title"]} — {match["heading"]}',
            match["content"],
            (
                f'Nguồn: {match["final_url"]} | Phiên bản: {match["version"]} '
                f'| Công bố: {match["publication_date"]} '
                f'| Mức tin cậy: {trust_label}'
            ),
        ])
        evidence.append({"id": evidence_id, "retrieved_at": retrieved_at, **match})
    lines.extend([
        "",
        SOURCE_DIFFERENCE_WARNING,
        "Đây là nội dung trích xuất từ tài liệu nguồn, không phải chẩn đoán "
        "hoặc chỉ định cho một bệnh nhân cụ thể.",
    ])
    audit_event("curated_guideline_retrieved", section_count=len(evidence))
    return {
        "status": "success",
        "response": "\n".join(lines),
        "evidence": evidence,
        "retrieved_at": retrieved_at,
    }


def retrieve_guidelines_with_auto_ingest(
    question: str,
    postgres_uri: str,
    endpoint: str,
    api_key: str,
    embedding_model: str,
    tavily_api_key: str,
    top_k: int = 3,
    min_score: float = 0.45,
    auto_ingest_enabled: bool = True,
    discovery_max_results: int = 5,
    auto_ingest_max_documents: int = 3,
    search_options: Optional[Dict[str, Any]] = None,
    embedder: Optional[Any] = None,
    on_date: Optional[date] = None,
    store: Optional[CuratedGuidelineStore] = None,
    searcher: Callable[..., Dict[str, Any]] = search_medical_guidelines,
    downloader: Callable[[str], DownloadedDocument] = download_approved_document,
) -> Dict[str, Any]:
    """Retrieve locally, then auto-index strict official documents on a miss."""
    if contains_sensitive_patient_data(question):
        return {"status": "privacy_denied", "response": SENSITIVE_SEARCH_REFUSAL}
    try:
        active_store = store or get_curated_guideline_store(postgres_uri)
        active_embedder = embedder or OpenAIEmbedder(
            endpoint, api_key, embedding_model
        )
    except (PostgresError, PoolTimeout, EmbeddingProviderError, ValueError) as error:
        logger.warning("Trusted guideline setup failed: %s", type(error).__name__)
        return {"status": "unavailable", "response": CURATED_RETRIEVAL_UNAVAILABLE}

    initial = retrieve_curated_guidelines(
        question=question,
        postgres_uri=postgres_uri,
        endpoint=endpoint,
        api_key=api_key,
        embedding_model=embedding_model,
        top_k=top_k,
        min_score=min_score,
        embedder=active_embedder,
        on_date=on_date,
        store=active_store,
    )
    if (
        not auto_ingest_enabled
        or initial.get("status") not in {"corpus_empty", "not_found"}
    ):
        return initial

    ingestion = auto_ingest_trusted_guidelines(
        question=question,
        store=active_store,
        embedder=active_embedder,
        tavily_api_key=tavily_api_key,
        discovery_max_results=discovery_max_results,
        max_documents=auto_ingest_max_documents,
        search_options=search_options,
        searcher=searcher,
        downloader=downloader,
        on_date=on_date,
    )
    if not ingestion["ingested"]:
        initial["auto_ingest"] = ingestion
        return initial

    result = retrieve_curated_guidelines(
        question=question,
        postgres_uri=postgres_uri,
        endpoint=endpoint,
        api_key=api_key,
        embedding_model=embedding_model,
        top_k=top_k,
        min_score=min_score,
        embedder=active_embedder,
        on_date=on_date,
        store=active_store,
    )
    result["auto_ingest"] = ingestion
    return result


def extract_sections(content: bytes, content_type: str) -> List[Tuple[str, str]]:
    """Extract full text while retaining an auditable section or page boundary."""
    normalized_type = content_type.split(";", 1)[0].strip().lower()
    if normalized_type in {"text/html", "application/xhtml+xml"}:
        parser = _SectionHTMLParser()
        parser.feed(content.decode("utf-8", errors="replace"))
        parser.close()
        return parser.sections
    if normalized_type == "text/plain":
        text = content.decode("utf-8", errors="replace")
        return [("Toàn văn", _normalize_extracted_text(text))]
    if normalized_type == "application/pdf":
        try:
            from pypdf import PdfReader  # pylint: disable=import-outside-toplevel
        except ImportError as error:
            raise GuidelineIngestionError(
                "PDF extraction requires the pypdf dependency"
            ) from error
        try:
            reader = PdfReader(BytesIO(content))
            return _extract_pdf_sections(reader)
        except Exception as error:  # pypdf exposes multiple parser exceptions
            raise GuidelineIngestionError("PDF full-text extraction failed") from error
    raise GuidelineIngestionError(f"Unsupported content type: {normalized_type}")


def chunk_sections(
    sections: Iterable[Tuple[str, str]],
    max_chars: int = 2200,
    overlap_chars: int = 200,
) -> List[Tuple[str, str]]:
    """Split inside each section only; never merge unrelated headings."""
    if max_chars < 200 or overlap_chars < 0 or overlap_chars >= max_chars:
        raise GuidelineIngestionError("Invalid section chunk settings")
    chunks = []
    for heading, raw_text in sections:
        text = _normalize_extracted_text(raw_text)
        if not text:
            continue
        start = 0
        part = 1
        while start < len(text):
            end = min(len(text), start + max_chars)
            if end < len(text):
                boundary = max(
                    text.rfind("\n\n", start + max_chars // 2, end),
                    text.rfind(". ", start + max_chars // 2, end),
                )
                if boundary > start:
                    end = boundary + (2 if text[boundary:boundary + 2] == ". " else 0)
            chunk = text[start:end].strip()
            if chunk:
                label = heading if start == 0 and end == len(text) else f"{heading} ({part})"
                chunks.append((label[:300], chunk))
            if end >= len(text):
                break
            next_start = max(start + 1, end - overlap_chars)
            while next_start < end and not text[next_start].isspace():
                next_start += 1
            start = next_start
            part += 1
    return chunks


class _SectionHTMLParser(HTMLParser):
    """Small HTML extractor that groups visible text under h1-h6 headings."""

    SKIP_TAGS = {"script", "style", "nav", "footer", "form", "svg", "noscript"}
    BREAK_TAGS = {
        "p", "li", "br", "tr", "td", "th", "div", "section", "article"
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.sections: List[Tuple[str, str]] = []
        self._heading = "Tổng quan"
        self._heading_buffer: List[str] = []
        self._text_buffer: List[str] = []
        self._heading_tag: Optional[str] = None
        self._skip_depth = 0

    def handle_starttag(self, tag: str, _attrs: List[Tuple[str, Optional[str]]]):
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if re.fullmatch(r"h[1-6]", tag):
            self._flush_section()
            self._heading_tag = tag
            self._heading_buffer = []
        elif tag in self.BREAK_TAGS:
            self._text_buffer.append("\n")

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if self._heading_tag == tag:
            heading = _normalize_extracted_text(" ".join(self._heading_buffer))
            if heading:
                self._heading = heading
            self._heading_tag = None
        elif tag in self.BREAK_TAGS:
            self._text_buffer.append("\n")

    def handle_data(self, data: str):
        if self._skip_depth:
            return
        if self._heading_tag:
            self._heading_buffer.append(data)
        else:
            self._text_buffer.append(data)

    def close(self):
        super().close()
        self._flush_section()

    def _flush_section(self):
        text = _normalize_extracted_text(" ".join(self._text_buffer))
        if text:
            self.sections.append((self._heading[:300], text))
        self._text_buffer = []


def _normalize_extracted_text(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n\n".join(line for line in lines if line)


def _extract_pdf_sections(reader: Any) -> List[Tuple[str, str]]:
    """Use PDF outline headings when present, otherwise retain page boundaries."""
    headings: Dict[int, str] = {}
    stack = list(reversed(getattr(reader, "outline", []) or []))
    visited = 0
    while stack and visited < 1000:
        item = stack.pop()
        visited += 1
        if isinstance(item, list):
            stack.extend(reversed(item))
            continue
        title = _normalize_extracted_text(str(getattr(item, "title", "")))
        if not title:
            continue
        try:
            page_number = int(reader.get_destination_page_number(item))
        except (AttributeError, TypeError, ValueError):
            continue
        headings.setdefault(page_number, title[:300])

    sections = []
    current_heading = ""
    for page_number, page in enumerate(reader.pages):
        current_heading = headings.get(page_number, current_heading)
        text = _normalize_extracted_text(page.extract_text() or "")
        if text:
            sections.append((
                current_heading or f"Trang {page_number + 1}",
                text,
            ))
    return sections


def _validate_metadata(metadata: DocumentMetadata) -> None:
    for name in ("title", "publisher", "version"):
        value = getattr(metadata, name)
        if not value or not value.strip() or len(value) > 300:
            raise GuidelineIngestionError(f"Invalid {name}")
    publication = _parse_iso_date(metadata.publication_date, "publication_date")
    effective_from = _parse_iso_date(metadata.effective_from, "effective_from")
    effective_until = None
    if metadata.effective_until:
        effective_until = _parse_iso_date(metadata.effective_until, "effective_until")
    if effective_until and effective_until < effective_from:
        raise GuidelineIngestionError("effective_until precedes effective_from")
    if publication > effective_from:
        raise GuidelineIngestionError("publication_date follows effective_from")


def _strict_publication_date(value: Any, today: date) -> Optional[str]:
    """Accept an explicit ISO date only; never invent source publication data."""
    text = str(value or "").strip()
    try:
        parsed = date.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00")).date()
        except ValueError:
            return None
    return parsed.isoformat() if parsed <= today else None


def _trusted_candidate_rank(candidate: Any) -> Tuple[int, float, str]:
    """Rank discovery candidates deterministically before full-document ingest."""
    if not isinstance(candidate, dict):
        return (999, 0.0, "")
    try:
        priority = int(candidate.get("source_priority", 999))
    except (TypeError, ValueError):
        priority = 999
    try:
        score = float(candidate.get("score", 0))
    except (TypeError, ValueError):
        score = 0.0
    url = str(candidate.get("final_url") or candidate.get("url") or "")
    return (priority, -score, url)


def _parse_iso_date(value: str, field: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise GuidelineIngestionError(f"{field} must use YYYY-MM-DD") from error


def _constant_time_equal(left: str, right: str) -> bool:
    return hmac.compare_digest(str(left), str(right))


def _validate_embedding_batch(
    sections: Sequence[Tuple[str, str]],
    embeddings: Sequence[Sequence[float]],
) -> None:
    if len(sections) != len(embeddings) or not sections:
        raise GuidelineIngestionError(
            "Every extracted section must have exactly one embedding"
        )
    dimensions = {len(vector) for vector in embeddings}
    if len(dimensions) != 1 or not dimensions or 0 in dimensions:
        raise GuidelineIngestionError("Embeddings must have one non-zero dimension")
    if any(
        not math.isfinite(float(value))
        for vector in embeddings
        for value in vector
    ):
        raise GuidelineIngestionError("Embeddings must contain finite numbers")


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return -1.0
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    if not all(math.isfinite(value) for value in left_values + right_values):
        return -1.0
    dot = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = math.sqrt(sum(value ** 2 for value in left_values))
    right_norm = math.sqrt(sum(value ** 2 for value in right_values))
    if left_norm == 0 or right_norm == 0:
        return -1.0
    return dot / (left_norm * right_norm)


def _host_has_only_public_addresses(
    host: str, resolver: Callable[..., Any]
) -> bool:
    try:
        records = resolver(host, 443, type=socket.SOCK_STREAM)
        addresses = {record[4][0] for record in records}
        return bool(addresses) and all(
            ipaddress.ip_address(address).is_global for address in addresses
        )
    except (OSError, ValueError, TypeError):
        return False
