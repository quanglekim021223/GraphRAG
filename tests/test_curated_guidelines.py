"""Tests for reviewed, versioned medical-guideline ingestion and retrieval."""
import os
import unittest
import uuid
from datetime import date, datetime, timezone

import psycopg
from psycopg import sql

from src.handlers.curated_guidelines import (
    CURATED_CORPUS_EMPTY,
    TRUSTED_OFFICIAL_STATUS,
    CuratedGuidelineStore,
    DocumentMetadata,
    DownloadedDocument,
    GuidelineIngestionError,
    _extract_pdf_sections,
    chunk_sections,
    download_approved_document,
    extract_sections,
    ingest_guideline,
    retrieve_curated_guidelines,
    retrieve_guidelines_with_auto_ingest,
)
from src.handlers.medical_guideline_search import SENSITIVE_SEARCH_REFUSAL


SOURCE_URL = "https://www.who.int/news-room/fact-sheets/detail/hypertension"


class _FakeResponse:
    def __init__(self, body=b"", status=200, url=SOURCE_URL, headers=None):
        self.body = body
        self.status = status
        self.url = url
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, limit):
        return self.body[:limit]

    def geturl(self):
        return self.url


def _public_resolver(_host, _port, type):  # pylint: disable=redefined-builtin
    return [(2, type, 6, "", ("93.184.216.34", 443))]


class _FakeEmbedder:
    model = "test-embedding-v1"

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            lowered = text.casefold()
            if "huyết áp" in lowered or "hypertension" in lowered:
                vectors.append([1.0, 0.0])
            elif "diabetes" in lowered:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.5, 0.5])
        return vectors


class _NonFiniteEmbedder(_FakeEmbedder):
    def embed(self, texts):
        self.calls.append(list(texts))
        return [[float("nan"), 1.0] for _text in texts]


class _FakePdfPage:
    def __init__(self, text):
        self.text = text

    def extract_text(self):
        return self.text


class _FakePdfDestination:
    def __init__(self, title, page_number):
        self.title = title
        self.page_number = page_number


class _FakePdfReader:
    def __init__(self):
        first = _FakePdfDestination("Recommendations", 0)
        second = _FakePdfDestination("Contraindications", 1)
        self.outline = [first, [second]]
        self.pages = [_FakePdfPage("Use when appropriate."), _FakePdfPage("Do not use.")]

    def get_destination_page_number(self, item):
        return item.page_number


def _metadata(version="2026.1", effective_until=None):
    return DocumentMetadata(
        title="WHO Hypertension Guideline",
        publisher="WHO",
        publication_date="2026-01-01",
        version=version,
        effective_from="2026-01-02",
        effective_until=effective_until,
    )


def _downloaded(content=b"Hypertension guidance"):
    return DownloadedDocument(
        source_url=SOURCE_URL,
        final_url=SOURCE_URL,
        content_type="text/html",
        content=content,
    )


class CuratedGuidelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_postgres_uri = os.getenv("TEST_POSTGRES_URI", "").strip()
        cls.schema_name = f"curated_guideline_test_{uuid.uuid4().hex}"
        cls.postgres_uri = ""
        if cls.base_postgres_uri:
            with psycopg.connect(
                cls.base_postgres_uri, autocommit=True
            ) as connection:
                connection.execute(
                    sql.SQL("CREATE SCHEMA {}").format(
                        sql.Identifier(cls.schema_name)
                    )
                )
            separator = "&" if "?" in cls.base_postgres_uri else "?"
            cls.postgres_uri = (
                f"{cls.base_postgres_uri}{separator}"
                f"options=-csearch_path%3D{cls.schema_name}"
            )

    @classmethod
    def tearDownClass(cls):
        if cls.base_postgres_uri:
            with psycopg.connect(
                cls.base_postgres_uri, autocommit=True
            ) as connection:
                connection.execute(
                    sql.SQL("DROP SCHEMA {} CASCADE").format(
                        sql.Identifier(cls.schema_name)
                    )
                )

    def setUp(self):
        self.embedder = _FakeEmbedder()
        self.store = None
        if self.postgres_uri:
            self.store = CuratedGuidelineStore(self.postgres_uri)
            with self.store.pool.connection() as connection:
                connection.execute("TRUNCATE guideline_documents CASCADE")

    def tearDown(self):
        if self.store is not None:
            self.store.close()

    def _require_store(self):
        if self.store is None:
            self.skipTest("TEST_POSTGRES_URI is required for PostgreSQL store tests")

    def _add_pending(self, version="2026.1", content=b"version one"):
        self._require_store()
        raw = (
            b"<h1>Hypertension</h1><p>" + content +
            b"</p><h2>Diabetes</h2><p>Monitor glucose.</p>"
        )
        return self.store.add_pending_document(
            downloaded=_downloaded(raw),
            metadata=_metadata(version),
            embedding_model=self.embedder.model,
            downloaded_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
        )

    def test_html_extraction_keeps_section_boundary_and_drops_scripts(self):
        sections = extract_sections(
            b"""
            <html><body><h1>Hypertension</h1>
            <p>Assess cardiovascular risk.</p>
            <script>malicious hidden text</script>
            <h2>Exceptions</h2><p>Review contraindications.</p></body></html>
            """,
            "text/html",
        )

        self.assertEqual("Hypertension", sections[0][0])
        self.assertIn("cardiovascular risk", sections[0][1])
        self.assertEqual("Exceptions", sections[1][0])
        self.assertNotIn("malicious", str(sections))

    def test_pdf_outline_is_preserved_as_section_heading(self):
        sections = _extract_pdf_sections(_FakePdfReader())

        self.assertEqual("Recommendations", sections[0][0])
        self.assertEqual("Contraindications", sections[1][0])

    def test_controlled_download_uses_validated_final_url_and_content_type(self):
        html = b"<h1>Approved guideline</h1><p>Full text</p>"

        document = download_approved_document(
            SOURCE_URL,
            head_opener=lambda request, timeout: _FakeResponse(
                status=200, url=request.full_url
            ),
            get_opener=lambda request, timeout: _FakeResponse(
                body=html,
                status=200,
                url=request.full_url,
                headers={"Content-Type": "text/html; charset=utf-8"},
            ),
            resolver=_public_resolver,
        )

        self.assertEqual(SOURCE_URL, document.final_url)
        self.assertEqual("text/html", document.content_type)
        self.assertEqual(html, document.content)

    def test_controlled_download_rejects_get_response_url_change(self):
        with self.assertRaisesRegex(GuidelineIngestionError, "changed URL"):
            download_approved_document(
                SOURCE_URL,
                head_opener=lambda request, timeout: _FakeResponse(
                    status=200, url=request.full_url
                ),
                get_opener=lambda _request, timeout: _FakeResponse(
                    body=b"untrusted",
                    status=200,
                    url="https://evil.example/document",
                    headers={"Content-Type": "text/html"},
                ),
                resolver=_public_resolver,
            )

    def test_chunking_never_merges_different_headings(self):
        sections = [
            ("Section A", "A sentence. " * 80),
            ("Section B", "B sentence. " * 80),
        ]

        chunks = chunk_sections(sections, max_chars=250, overlap_chars=25)

        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(
            not ("A sentence" in text and "B sentence" in text)
            for _heading, text in chunks
        ))

    def test_ingestion_is_pending_until_reviewed_hash_is_approved(self):
        self._require_store()
        html = (
            b"<h1>Blood pressure treatment</h1>"
            b"<p>Hypertension requires cardiovascular risk assessment.</p>"
        )

        document = ingest_guideline(
            store=self.store,
            source_url=SOURCE_URL,
            metadata=_metadata(),
            embedding_model=self.embedder.model,
            downloader=lambda _url: _downloaded(html),
        )

        self.assertEqual("pending_review", document["review_status"])
        self.assertEqual(0, document["section_count"])
        self.assertEqual([], self.embedder.calls)
        review_bundle = self.store.get_review_bundle(document["document_id"])
        self.assertIn("cardiovascular risk", review_bundle["sections"][0]["content"])
        pending_result = retrieve_curated_guidelines(
            "Hướng dẫn tăng huyết áp",
            self.postgres_uri,
            "unused",
            "unused",
            self.embedder.model,
            embedder=self.embedder,
            on_date=date(2026, 1, 10),
            store=self.store,
        )
        self.assertEqual("corpus_empty", pending_result["status"])
        self.assertEqual(CURATED_CORPUS_EMPTY, pending_result["response"])

        with self.assertRaises(GuidelineIngestionError):
            self.store.approve(
                document["document_id"],
                "reviewer-1",
                "wrong-hash",
                self.embedder,
            )
        self.assertEqual([], self.embedder.calls)

        approved = self.store.approve(
            document["document_id"],
            "reviewer-1",
            document["content_hash"],
            self.embedder,
        )
        self.assertEqual("approved", approved["review_status"])
        self.assertEqual("active", approved["effective_status"])

        result = retrieve_curated_guidelines(
            "Hướng dẫn tăng huyết áp",
            self.postgres_uri,
            "unused",
            "unused",
            self.embedder.model,
            top_k=1,
            min_score=0.8,
            embedder=self.embedder,
            on_date=date(2026, 1, 10),
            store=self.store,
        )

        self.assertEqual("success", result["status"])
        self.assertIn("[G1]", result["response"])
        self.assertIn("Blood pressure treatment", result["response"])
        self.assertIn("Phiên bản: 2026.1", result["response"])
        self.assertEqual(document["content_hash"], result["evidence"][0]["document_hash"])
        self.assertEqual(64, len(result["evidence"][0]["section_hash"]))

    def test_approving_new_version_supersedes_old_version(self):
        first = self._add_pending("2026.1", b"first")
        self.store.approve(
            first["document_id"], "reviewer", first["content_hash"], self.embedder
        )
        second = self._add_pending("2026.2", b"second")
        self.store.approve(
            second["document_id"], "reviewer", second["content_hash"], self.embedder
        )

        documents = {
            item["version"]: item for item in self.store.list_documents()
        }

        self.assertEqual("superseded", documents["2026.1"]["effective_status"])
        self.assertEqual("active", documents["2026.2"]["effective_status"])
        matches = self.store.search(
            [1.0, 0.0], self.embedder.model, 5, 0.8, date(2026, 1, 10)
        )
        self.assertTrue(matches)
        self.assertTrue(all(item["version"] == "2026.2" for item in matches))

    def test_corpus_miss_auto_ingests_full_trusted_official_document(self):
        self._require_store()
        search_calls = []
        download_calls = []

        def searcher(**kwargs):
            search_calls.append(kwargs)
            return {
                "status": "success",
                "evidence": [{
                    "title": "WHO hypertension guidance",
                    "final_url": SOURCE_URL,
                    "source_name": "WHO",
                    "publication_date": "2026-01-01",
                }],
            }

        def downloader(url):
            download_calls.append(url)
            return _downloaded(
                b"<h1>Hypertension treatment</h1>"
                b"<p>Assess cardiovascular risk before treatment.</p>"
            )

        result = retrieve_guidelines_with_auto_ingest(
            question="Hướng dẫn tăng huyết áp",
            postgres_uri=self.postgres_uri,
            endpoint="unused",
            api_key="unused",
            embedding_model=self.embedder.model,
            tavily_api_key="tavily-key",
            top_k=1,
            min_score=0.8,
            embedder=self.embedder,
            on_date=date(2026, 1, 10),
            store=self.store,
            searcher=searcher,
            downloader=downloader,
        )

        self.assertEqual("success", result["status"])
        self.assertEqual(1, len(result["auto_ingest"]["ingested"]))
        self.assertEqual(TRUSTED_OFFICIAL_STATUS,
                         result["evidence"][0]["review_status"])
        self.assertIn("Nguồn chính thức tự động xác minh", result["response"])
        self.assertEqual(1, len(search_calls))
        self.assertEqual([SOURCE_URL], download_calls)
        self.assertEqual(
            TRUSTED_OFFICIAL_STATUS,
            self.store.list_documents()[0]["review_status"],
        )

    def test_auto_ingest_rejects_candidate_without_publication_date(self):
        self._require_store()
        download_calls = []

        result = retrieve_guidelines_with_auto_ingest(
            question="Hướng dẫn tăng huyết áp",
            postgres_uri=self.postgres_uri,
            endpoint="unused",
            api_key="unused",
            embedding_model=self.embedder.model,
            tavily_api_key="tavily-key",
            embedder=self.embedder,
            on_date=date(2026, 1, 10),
            store=self.store,
            searcher=lambda **_kwargs: {
                "status": "success",
                "evidence": [{
                    "title": "WHO hypertension guidance",
                    "final_url": SOURCE_URL,
                    "source_name": "WHO",
                    "publication_date": None,
                }],
            },
            downloader=lambda url: download_calls.append(url),
        )

        self.assertEqual("corpus_empty", result["status"])
        self.assertEqual("not_found", result["auto_ingest"]["status"])
        self.assertEqual(1, result["auto_ingest"]["skipped"])
        self.assertEqual([], download_calls)

    def test_expired_or_withdrawn_document_is_not_retrievable(self):
        self._require_store()
        expired_metadata = _metadata(effective_until="2026-01-05")
        expired = self.store.add_pending_document(
            _downloaded(),
            expired_metadata,
            self.embedder.model,
        )
        self.store.approve(
            expired["document_id"],
            "reviewer",
            expired["content_hash"],
            self.embedder,
        )
        self.assertFalse(
            self.store.has_effective_documents(
                self.embedder.model, date(2026, 1, 10)
            )
        )

        active = self._add_pending("2026.2", b"active")
        self.store.approve(
            active["document_id"],
            "reviewer",
            active["content_hash"],
            self.embedder,
        )
        self.store.withdraw(active["document_id"], "reviewer")
        self.assertFalse(
            self.store.has_effective_documents(
                self.embedder.model, date(2026, 1, 10)
            )
        )

    def test_same_url_and_version_are_immutable(self):
        self._add_pending()

        with self.assertRaisesRegex(GuidelineIngestionError, "immutable"):
            self._add_pending(content=b"changed content")

    def test_non_finite_embedding_is_rejected(self):
        document = self._add_pending()
        with self.assertRaisesRegex(GuidelineIngestionError, "finite"):
            self.store.approve(
                document["document_id"],
                "reviewer",
                document["content_hash"],
                _NonFiniteEmbedder(),
            )
        self.assertEqual(
            "pending_review",
            self.store.get_document(document["document_id"])["review_status"],
        )

    def test_sensitive_question_is_rejected_before_database_or_embedding(self):
        result = retrieve_curated_guidelines(
            "patient_id P-123 đang dùng thuốc gì?",
            "postgresql://unused:unused@127.0.0.1:1/unused",
            "unused",
            "unused",
            self.embedder.model,
            embedder=self.embedder,
        )

        self.assertEqual("privacy_denied", result["status"])
        self.assertEqual(SENSITIVE_SEARCH_REFUSAL, result["response"])
        self.assertEqual([], self.embedder.calls)

    def test_sensitive_question_never_reaches_auto_ingest_search(self):
        search_calls = []

        result = retrieve_guidelines_with_auto_ingest(
            question="patient_id P-123 đang dùng thuốc gì?",
            postgres_uri="postgresql://unused:unused@127.0.0.1:1/unused",
            endpoint="unused",
            api_key="unused",
            embedding_model=self.embedder.model,
            tavily_api_key="tavily-key",
            embedder=self.embedder,
            searcher=lambda **kwargs: search_calls.append(kwargs),
        )

        self.assertEqual("privacy_denied", result["status"])
        self.assertEqual(SENSITIVE_SEARCH_REFUSAL, result["response"])
        self.assertEqual([], search_calls)
        self.assertEqual([], self.embedder.calls)


if __name__ == "__main__":
    unittest.main()
