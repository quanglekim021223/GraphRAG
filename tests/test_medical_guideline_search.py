"""Tests for allowlisted, privacy-safe medical guideline retrieval."""
import json
import unittest
from urllib.error import HTTPError, URLError

from src.handlers.medical_guideline_search import (
    BUDGET_EXHAUSTED_RESPONSE,
    CIRCUIT_OPEN_RESPONSE,
    NO_GUIDELINE_RESULT,
    RATE_LIMIT_RESPONSE,
    SEARCH_UNAVAILABLE,
    SENSITIVE_SEARCH_REFUSAL,
    _reset_runtime_state_for_tests,
    contains_sensitive_patient_data,
    is_approved_source_url,
    resolve_approved_final_url,
    search_medical_guidelines,
)


class _FakeResponse:
    def __init__(self, body=None, status=200, url="", headers=None):
        self.body = json.dumps(body or {}).encode("utf-8")
        self.status = status
        self.url = url
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit):
        return self.body

    def geturl(self):
        return self.url


def _public_resolver(_host, _port, type):  # pylint: disable=redefined-builtin
    return [(2, type, 6, "", ("93.184.216.34", 443))]


def _approved_source_opener(request, timeout):
    if timeout != 5:
        raise AssertionError("unexpected source timeout")
    return _FakeResponse(status=200, url=request.full_url)


class MedicalGuidelineSearchTests(unittest.TestCase):
    def setUp(self):
        _reset_runtime_state_for_tests()

    def _search(self, question, opener, **kwargs):
        return search_medical_guidelines(
            question,
            "secret",
            opener=opener,
            source_opener=_approved_source_opener,
            resolver=_public_resolver,
            cache_ttl_seconds=0,
            **kwargs,
        )

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
        self.assertTrue(contains_sensitive_patient_data("Huyết áp của Alice là gì?"))
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

        result = self._search(
            "Hướng dẫn chung về tăng huyết áp",
            opener,
        )

        self.assertEqual("success", result["status"])
        self.assertEqual(1, len(result["evidence"]))
        self.assertIn("[S1] WHO: WHO hypertension fact sheet", result["response"])
        self.assertIn("Approved source content.", result["response"])
        self.assertNotIn("Spoofed", result["response"])
        self.assertEqual(
            result["evidence"][0]["url"],
            result["evidence"][0]["final_url"],
        )
        self.assertEqual(64, len(result["evidence"][0]["content_hash"]))
        self.assertIn("retrieved_at", result["evidence"][0])
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

        result = self._search("Huyết áp là gì?", opener)

        self.assertEqual("not_found", result["status"])
        self.assertEqual(NO_GUIDELINE_RESULT, result["response"])

    def test_redirect_chain_accepts_only_approved_final_url(self):
        start = "https://www.who.int/publications/i/item/start"
        calls = []

        def opener(request, timeout):
            self.assertEqual(5, timeout)
            calls.append(request.full_url)
            if len(calls) == 1:
                return _FakeResponse(
                    status=302,
                    url=request.full_url,
                    headers={"Location": "/publications/i/item/final"},
                )
            return _FakeResponse(status=200, url=request.full_url)

        final_url = resolve_approved_final_url(
            start, opener=opener, resolver=_public_resolver
        )

        self.assertEqual(
            "https://www.who.int/publications/i/item/final", final_url
        )
        self.assertEqual(2, len(calls))

    def test_redirect_to_unapproved_host_is_rejected_before_following(self):
        calls = []

        def opener(request, timeout):
            self.assertEqual(5, timeout)
            calls.append(request.full_url)
            return _FakeResponse(
                status=302,
                url=request.full_url,
                headers={"Location": "https://evil.example/guideline"},
            )

        final_url = resolve_approved_final_url(
            "https://www.who.int/publications/i/item/start",
            opener=opener,
            resolver=_public_resolver,
        )

        self.assertIsNone(final_url)
        self.assertEqual(1, len(calls))

    def test_private_dns_target_is_rejected_before_request(self):
        calls = []

        def private_resolver(_host, _port, type):  # pylint: disable=redefined-builtin
            return [(2, type, 6, "", ("127.0.0.1", 443))]

        final_url = resolve_approved_final_url(
            "https://www.who.int/publications/i/item/start",
            opener=lambda *args, **kwargs: calls.append((args, kwargs)),
            resolver=private_resolver,
        )

        self.assertIsNone(final_url)
        self.assertEqual([], calls)

    def test_retry_respects_short_retry_after(self):
        calls = []
        sleeps = []

        def opener(request, timeout):
            self.assertEqual(10, timeout)
            calls.append(request)
            if len(calls) == 1:
                raise HTTPError(
                    request.full_url, 429, "rate limited", {"Retry-After": "1"}, None
                )
            return _FakeResponse({"results": []})

        result = self._search(
            "Huyết áp là gì?",
            opener,
            max_retries=1,
            max_retry_delay_seconds=2,
            sleeper=sleeps.append,
        )

        self.assertEqual("not_found", result["status"])
        self.assertEqual(2, len(calls))
        self.assertEqual([1.0], sleeps)

    def test_retry_after_above_cap_fails_without_sleeping(self):
        sleeps = []

        def opener(request, timeout):
            self.assertEqual(10, timeout)
            raise HTTPError(
                request.full_url, 429, "rate limited", {"Retry-After": "60"}, None
            )

        result = self._search(
            "Huyết áp là gì?",
            opener,
            max_retries=1,
            max_retry_delay_seconds=2,
            sleeper=sleeps.append,
        )

        self.assertEqual("unavailable", result["status"])
        self.assertEqual([], sleeps)

    def test_retry_does_not_exceed_daily_call_budget(self):
        calls = []

        def opener(request, timeout):
            self.assertEqual(10, timeout)
            calls.append(1)
            raise HTTPError(
                request.full_url, 503, "unavailable", {}, None
            )

        result = self._search(
            "Huyết áp là gì?",
            opener,
            daily_budget=1,
            max_retries=1,
            sleeper=lambda _delay: self.fail("retry should not sleep"),
        )

        self.assertEqual("budget_exhausted", result["status"])
        self.assertEqual(1, len(calls))

    def test_sources_are_ordered_by_approved_priority_then_score(self):
        body = {"results": [
            {
                "title": "CDC guidance",
                "url": "https://www.cdc.gov/flu/hcp/clinical-guidance/index.html",
                "content": "CDC content",
                "score": 0.99,
            },
            {
                "title": "Ministry guidance",
                "url": "https://emohbackup.moh.gov.vn/publish/attach/getfile/1",
                "content": "Ministry content",
                "score": 0.7,
            },
            {
                "title": "WHO guidance",
                "url": "https://www.who.int/publications/i/item/guide",
                "content": "WHO content",
                "score": 0.8,
            },
        ]}

        result = self._search(
            "Hướng dẫn y khoa tổng quát",
            lambda _request, timeout: _FakeResponse(body),
        )

        self.assertEqual("success", result["status"])
        self.assertEqual(
            ["Bộ Y tế Việt Nam", "WHO", "CDC"],
            [item["source_name"] for item in result["evidence"]],
        )
        self.assertIn("không tự kết luận", result["response"])

    def test_cache_avoids_second_provider_call(self):
        calls = []
        body = {"results": [{
            "title": "WHO guidance",
            "url": "https://www.who.int/publications/i/item/guide",
            "content": "Approved content",
            "score": 0.9,
        }]}

        def opener(_request, timeout):
            self.assertEqual(10, timeout)
            calls.append(1)
            return _FakeResponse(body)

        first = search_medical_guidelines(
            "Hướng dẫn tăng huyết áp",
            "secret",
            opener=opener,
            source_opener=_approved_source_opener,
            resolver=_public_resolver,
            cache_ttl_seconds=300,
        )
        second = search_medical_guidelines(
            "  hướng dẫn  TĂNG huyết áp ",
            "secret",
            opener=lambda *_args, **_kwargs: self.fail("cache missed"),
            source_opener=_approved_source_opener,
            resolver=_public_resolver,
            cache_ttl_seconds=300,
        )

        self.assertEqual("success", first["status"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(1, len(calls))

    def test_rate_limit_blocks_second_uncached_call(self):
        calls = []

        def opener(_request, timeout):
            self.assertEqual(10, timeout)
            calls.append(1)
            return _FakeResponse({"results": []})

        first = self._search(
            "Câu hỏi y khoa thứ nhất", opener, rate_limit_per_minute=1
        )
        second = self._search(
            "Câu hỏi y khoa thứ hai", opener, rate_limit_per_minute=1
        )

        self.assertEqual("not_found", first["status"])
        self.assertEqual("rate_limited", second["status"])
        self.assertEqual(RATE_LIMIT_RESPONSE, second["response"])
        self.assertEqual(1, len(calls))

    def test_daily_budget_blocks_second_provider_call(self):
        calls = []

        def opener(_request, timeout):
            self.assertEqual(10, timeout)
            calls.append(1)
            return _FakeResponse({"results": []})

        first = self._search(
            "Câu hỏi y khoa thứ nhất", opener, daily_budget=1
        )
        second = self._search(
            "Câu hỏi y khoa thứ hai", opener, daily_budget=1
        )

        self.assertEqual("not_found", first["status"])
        self.assertEqual("budget_exhausted", second["status"])
        self.assertEqual(BUDGET_EXHAUSTED_RESPONSE, second["response"])
        self.assertEqual(1, len(calls))

    def test_circuit_breaker_opens_after_configured_failures(self):
        calls = []

        def failing_opener(_request, timeout):
            self.assertEqual(10, timeout)
            calls.append(1)
            raise URLError("down")

        first = self._search(
            "Câu hỏi y khoa thứ nhất",
            failing_opener,
            max_retries=0,
            circuit_failure_threshold=1,
        )
        second = self._search(
            "Câu hỏi y khoa thứ hai",
            failing_opener,
            max_retries=0,
            circuit_failure_threshold=1,
        )

        self.assertEqual("unavailable", first["status"])
        self.assertEqual("circuit_open", second["status"])
        self.assertEqual(CIRCUIT_OPEN_RESPONSE, second["response"])
        self.assertEqual(1, len(calls))


if __name__ == "__main__":
    unittest.main()
