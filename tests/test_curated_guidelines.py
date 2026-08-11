"""Tests for reviewed, versioned medical-guideline ingestion and retrieval."""
import os
import unittest
import uuid
from datetime import date, datetime, timezone

import psycopg
from psycopg import sql

from src.handlers.curated_guidelines import (
    CURATED_CORPUS_EMPTY,
    DEFAULT_PREWARM_TOPICS,
    TRUSTED_OFFICIAL_STATUS,
    CuratedGuidelineStore,
    DocumentMetadata,
    DownloadedDocument,
    ExtractedSection,
    GuidelineIngestionError,
    _extract_pdf_document_sections,
    _extract_pdf_sections,
    build_child_chunks,
    chunk_sections,
    download_approved_document,
    extract_sections,
    extract_document_sections,
    ingest_guideline,
    prewarm_guideline_corpus,
    retrieve_curated_guidelines,
    retrieve_guidelines_with_auto_ingest,
)
from src.handlers.medical_guideline_search import SENSITIVE_SEARCH_REFUSAL


SOURCE_URL = "https://www.who.int/news-room/fact-sheets/detail/hypertension"
WHO_DIABETES_URL = "https://www.who.int/news-room/fact-sheets/detail/diabetes"
WHO_OBESITY_URL = (
    "https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight"
)
NICE_URL = "https://www.nice.org.uk/guidance/ng136"
CDC_URL = "https://www.cdc.gov/flu/hcp/clinical-guidance/index.html"


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

    def test_prewarm_deduplicates_topics_and_reports_coverage(self):
        calls = []

        def retriever(question, **options):
            calls.append((question, options))
            if question == "Hypertension guideline":
                return {"status": "success", "evidence": [{"id": "G1"}]}
            if question == "Diabetes guideline":
                return {
                    "status": "success",
                    "evidence": [{"id": "G1"}],
                    "auto_ingest": {
                        "status": "success",
                        "ingested": [{"document_id": "doc-1"}],
                    },
                }
            return {
                "status": "not_found",
                "auto_ingest": {"status": "not_found", "ingested": []},
            }

        result = prewarm_guideline_corpus(
            topics=[
                " Hypertension   guideline ",
                "hypertension guideline",
                "Diabetes guideline",
                "Rare disease guideline",
            ],
            retrieval_options={"auto_ingest_enabled": False, "marker": "test"},
            retriever=retriever,
        )

        self.assertEqual("partial", result["status"])
        self.assertEqual(3, result["topics_total"])
        self.assertEqual(1, result["already_covered"])
        self.assertEqual(1, result["warmed"])
        self.assertEqual(1, result["not_covered"])
        self.assertEqual(1, result["documents_ingested"])
        self.assertTrue(all(
            options["auto_ingest_enabled"] for _topic, options in calls
        ))

    def test_prewarm_stops_after_terminal_discovery_failure(self):
        calls = []

        def retriever(question, **_options):
            calls.append(question)
            return {
                "status": "corpus_empty",
                "auto_ingest": {"status": "budget_exhausted", "ingested": []},
            }

        result = prewarm_guideline_corpus(
            topics=["Topic one", "Topic two"],
            retrieval_options={},
            retriever=retriever,
        )

        self.assertEqual("failed", result["status"])
        self.assertEqual(1, result["topics_processed"])
        self.assertFalse(result["completed_all"])
        self.assertEqual("budget_exhausted", result["stopped_reason"])
        self.assertEqual(["Topic one"], calls)

    def test_default_prewarm_taxonomy_is_bounded_and_deidentified(self):
        self.assertEqual(10, len(DEFAULT_PREWARM_TOPICS))
        self.assertEqual(len(DEFAULT_PREWARM_TOPICS), len(set(DEFAULT_PREWARM_TOPICS)))
        self.assertTrue(all(topic.strip() for topic in DEFAULT_PREWARM_TOPICS))

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
        self.assertEqual("Hypertension > Exceptions", sections[1][0])
        self.assertNotIn("malicious", str(sections))

    def test_html_extraction_preserves_heading_hierarchy(self):
        sections = extract_document_sections(
            b"""
            <h1>Diabetes</h1><p>Overview.</p>
            <h2>Treatment</h2><p>Use treatment.</p>
            <h3>Contraindications</h3><p>Avoid when unsafe.</p>
            """,
            "text/html",
        )

        self.assertEqual("Diabetes", sections[0].section_path)
        self.assertEqual("Diabetes > Treatment", sections[1].section_path)
        self.assertEqual(
            "Diabetes > Treatment > Contraindications",
            sections[2].section_path,
        )

    def test_plain_text_uses_conservative_heading_hierarchy(self):
        sections = extract_document_sections(
            b"# Diabetes\nOverview.\n\n## Treatment\nUse treatment.\n",
            "text/plain",
        )

        self.assertEqual("Diabetes", sections[0].section_path)
        self.assertEqual("Diabetes > Treatment", sections[1].section_path)

        numbered = extract_document_sections(
            b"1 Overview\nGeneral guidance.\n\n1.1 Treatment\nUse treatment.\n",
            "text/plain",
        )
        self.assertEqual("1 Overview", numbered[0].section_path)
        self.assertEqual("1 Overview > 1.1 Treatment", numbered[1].section_path)

    def test_pdf_outline_is_preserved_as_section_heading(self):
        sections = _extract_pdf_sections(_FakePdfReader())
        structured = _extract_pdf_document_sections(_FakePdfReader())

        self.assertEqual("Recommendations", sections[0][0])
        self.assertEqual("Contraindications", sections[1][0])
        self.assertEqual(
            "Recommendations > Contraindications",
            structured[1].section_path,
        )

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

    def test_child_chunking_prepends_title_and_path_with_token_budget(self):
        sections = [ExtractedSection(
            heading="Contraindications",
            section_path="Diabetes > Treatment > Contraindications",
            level=3,
            content=(
                "First complete recommendation is retained. "
                "Second complete recommendation is retained. " * 20
            ),
        )]

        chunks = build_child_chunks(
            sections,
            document_title="Clinical Guideline",
            max_tokens=96,
            overlap_tokens=8,
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.token_count <= 96 for chunk in chunks))
        self.assertTrue(all(
            chunk.embedding_text.startswith(
                "Document: Clinical Guideline\n"
                "Section: Diabetes > Treatment > Contraindications"
            )
            for chunk in chunks
        ))

    def test_single_oversized_sentence_is_hard_split_within_budget(self):
        chunks = build_child_chunks(
            [ExtractedSection(
                heading="Long sentence",
                section_path="Long sentence",
                level=1,
                content="word " * 300,
            )],
            document_title="Guideline",
            max_tokens=80,
            overlap_tokens=5,
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.token_count <= 80 for chunk in chunks))

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
        self.assertEqual(1, approved["parent_section_count"])
        with self.store.pool.connection() as connection:
            relation = connection.execute(
                """
                SELECT p.section_path, s.chunk_index, s.token_count
                FROM guideline_sections s
                JOIN guideline_parent_sections p
                  ON p.parent_section_id = s.parent_section_id
                WHERE s.document_id = %s
                """,
                (document["document_id"],),
            ).fetchone()
        self.assertEqual("Blood pressure treatment", relation["section_path"])
        self.assertEqual(0, relation["chunk_index"])
        self.assertGreater(relation["token_count"], 0)

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
        self.assertEqual("parent", result["evidence"][0]["context_mode"])

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

    def test_retrieval_expands_neighbor_chunks_for_large_parent(self):
        self._require_store()
        content = " ".join(
            f"Clinical recommendation number {index} is retained."
            for index in range(80)
        )
        document = self.store.add_pending_document(
            _downloaded(
                f"<h1>Long recommendations</h1><p>{content}</p>".encode()
            ),
            _metadata(),
            self.embedder.model,
        )
        approved = self.store.approve(
            document["document_id"],
            "reviewer",
            document["content_hash"],
            self.embedder,
            max_chunk_tokens=80,
            chunk_overlap_tokens=5,
        )

        self.assertGreater(approved["section_count"], 2)
        matches = self.store.search(
            [1.0, 0.0],
            self.embedder.model,
            top_k=1,
            min_score=0.0,
            on_date=date(2026, 1, 10),
            neighbor_window=1,
            parent_context_max_tokens=200,
        )

        self.assertEqual("neighbors", matches[0]["context_mode"])
        self.assertIn(len(matches[0]["context_section_ids"]), {2, 3})
        self.assertIn(matches[0]["section_id"], matches[0]["context_section_ids"])
        self.assertTrue(
            matches[0]["previous_section_id"]
            or matches[0]["next_section_id"]
        )

    def test_corpus_miss_auto_ingests_full_trusted_official_document(self):
        self._require_store()
        search_calls = []
        download_calls = []

        def searcher(**kwargs):
            search_calls.append(kwargs)
            return {
                "status": "success",
                "evidence": [
                    {
                        "title": "CDC influenza guidance",
                        "final_url": CDC_URL,
                        "source_name": "CDC",
                        "publication_date": "2026-01-01",
                        "source_priority": 40,
                        "score": 0.99,
                    },
                    {
                        "title": "WHO obesity guidance",
                        "final_url": WHO_OBESITY_URL,
                        "source_name": "WHO",
                        "publication_date": "2026-01-01",
                        "source_priority": 20,
                        "score": 0.80,
                    },
                    {
                        "title": "WHO hypertension without a date",
                        "final_url": SOURCE_URL,
                        "source_name": "WHO",
                        "publication_date": None,
                        "source_priority": 20,
                        "score": 0.99,
                    },
                    {
                        "title": "NICE hypertension guidance",
                        "final_url": NICE_URL,
                        "source_name": "NICE",
                        "publication_date": "2026-01-01",
                        "source_priority": 30,
                        "score": 0.90,
                    },
                    {
                        "title": "WHO diabetes guidance",
                        "final_url": WHO_DIABETES_URL,
                        "source_name": "WHO",
                        "publication_date": "2026-01-01",
                        "source_priority": 20,
                        "score": 0.95,
                    },
                ],
            }

        def downloader(url):
            download_calls.append(url)
            return DownloadedDocument(
                source_url=url,
                final_url=url,
                content_type="text/html",
                content=(
                    "<h1>Hypertension treatment</h1>"
                    f"<p>Official document: {url}</p>"
                ).encode("utf-8"),
            )

        result = retrieve_guidelines_with_auto_ingest(
            question="Hướng dẫn tăng huyết áp",
            postgres_uri=self.postgres_uri,
            endpoint="unused",
            api_key="unused",
            embedding_model=self.embedder.model,
            tavily_api_key="tavily-key",
            top_k=3,
            min_score=0.8,
            embedder=self.embedder,
            on_date=date(2026, 1, 10),
            store=self.store,
            searcher=searcher,
            downloader=downloader,
        )

        self.assertEqual("success", result["status"])
        self.assertEqual(5, search_calls[0]["max_results"])
        self.assertEqual(5, result["auto_ingest"]["discovered"])
        self.assertEqual(3, len(result["auto_ingest"]["ingested"]))
        self.assertEqual(1, result["auto_ingest"]["skipped"])
        self.assertEqual(3, len(result["evidence"]))
        self.assertTrue(all(
            item["review_status"] == TRUSTED_OFFICIAL_STATUS
            for item in result["evidence"]
        ))
        self.assertIn("Nguồn chính thức tự động xác minh", result["response"])
        self.assertEqual(1, len(search_calls))
        self.assertEqual(
            [WHO_DIABETES_URL, WHO_OBESITY_URL, NICE_URL], download_calls
        )
        self.assertEqual(3, len(self.store.list_documents()))
        self.assertTrue(all(
            item["review_status"] == TRUSTED_OFFICIAL_STATUS
            for item in self.store.list_documents()
        ))

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
