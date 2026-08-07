"""Tests for allowlisted, privacy-safe medical guideline retrieval."""
import json
import unittest

from src.handlers.medical_guideline_search import (
    NO_GUIDELINE_RESULT,
    SEARCH_UNAVAILABLE,
    SENSITIVE_SEARCH_REFUSAL,
    contains_sensitive_patient_data,
    is_approved_source_url,
    search_medical_guidelines,
)


class _FakeResponse:
    def __init__(self, body):
        self.body = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit):
        return self.body


class MedicalGuidelineSearchTests(unittest.TestCase):
    def test_url_requires_exact_host_https_and_approved_path(self):
        self.assertTrue(is_approved_source_url(
            "https://www.who.int/news-room/fact-sheets/detail/hypertension"
        ))
        self.assertTrue(is_approved_source_url(
            "https://www.nice.org.uk/guidance/ng136"
        ))
        self.assertTrue(is_approved_source_url(
            "https://www.cdc.gov/flu/hcp/clinical-guidance/index.html"
        ))
        self.assertFalse(is_approved_source_url(
            "https://who.int.evil.example/news-room/fact-sheets/detail/x"
        ))
        self.assertFalse(is_approved_source_url(
            "http://www.who.int/news-room/fact-sheets/detail/x"
        ))
        self.assertFalse(is_approved_source_url(
            "https://www.who.int/about/policies"
        ))

    def test_sensitive_patient_query_never_calls_search_provider(self):
        calls = []

        def opener(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("network must not be called")

        result = search_medical_guidelines(
            "patient_id: P-123 bị bệnh gì?", "secret", opener=opener
        )

        self.assertTrue(contains_sensitive_patient_data("patient_id: P-123"))
        self.assertTrue(contains_sensitive_patient_data("Alice huyết áp bao nhiêu?"))
        self.assertEqual("privacy_denied", result["status"])
        self.assertEqual(SENSITIVE_SEARCH_REFUSAL, result["response"])
        self.assertEqual([], calls)

    def test_missing_api_key_fails_closed_without_network(self):
        result = search_medical_guidelines("Huyết áp là gì?", "")

        self.assertEqual("unavailable", result["status"])
        self.assertEqual(SEARCH_UNAVAILABLE, result["response"])

    def test_search_filters_provider_results_and_renders_citations(self):
        captured = {}
        provider_body = {
            "results": [
                {
                    "title": "Spoofed WHO",
                    "url": "https://who.int.evil.example/guidance",
                    "content": "untrusted",
                    "score": 0.99,
                },
                {
                    "title": "WHO hypertension fact sheet",
                    "url": (
                        "https://www.who.int/news-room/fact-sheets/detail/"
                        "hypertension"
                    ),
                    "content": "  Approved\nsource\tcontent.  ",
                    "score": 0.91,
                },
                {
                    "title": "Low score",
                    "url": "https://www.nice.org.uk/guidance/ng136",
                    "content": "low relevance",
                    "score": 0.2,
                },
            ]
        }

        def opener(request, timeout):
            captured["timeout"] = timeout
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["authorization"] = request.get_header("Authorization")
            return _FakeResponse(provider_body)

        result = search_medical_guidelines(
            "Hướng dẫn chung về tăng huyết áp",
            "secret",
            opener=opener,
        )

        self.assertEqual("success", result["status"])
        self.assertEqual(1, len(result["evidence"]))
        self.assertIn("[S1] WHO hypertension fact sheet", result["response"])
        self.assertIn("Approved source content.", result["response"])
        self.assertNotIn("Spoofed", result["response"])
        self.assertFalse(captured["payload"]["include_answer"])
        self.assertFalse(captured["payload"]["include_raw_content"])
        self.assertIn("who.int", captured["payload"]["include_domains"])
        self.assertEqual("Bearer secret", captured["authorization"])
        self.assertEqual(10, captured["timeout"])

    def test_only_unapproved_results_abstains(self):
        def opener(_request, timeout):
            self.assertEqual(10, timeout)
            return _FakeResponse({"results": [{
                "title": "Blog",
                "url": "https://example.com/medical-advice",
                "content": "unsupported",
                "score": 1,
            }]})

        result = search_medical_guidelines(
            "Huyết áp là gì?", "secret", opener=opener
        )

        self.assertEqual("not_found", result["status"])
        self.assertEqual(NO_GUIDELINE_RESULT, result["response"])


if __name__ == "__main__":
    unittest.main()
