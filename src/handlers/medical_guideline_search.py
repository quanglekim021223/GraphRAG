"""Allowlisted, retrieval-only medical guideline search.

The search provider is used only to retrieve snippets. The application disables
provider-generated answers, post-filters every result URL, and returns the
allowlisted snippets with citations. Patient-identifying queries are
rejected before any network request.
"""
import copy
import hashlib
import ipaddress
import json
import re
import socket
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from threading import Lock
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

from src.handlers.security_guardrails import audit_event
from src.helpers.logging_config import logger


TAVILY_SEARCH_URL = "https://api.tavily.com/search"
MAX_RESPONSE_BYTES = 1_000_000
MAX_QUESTION_LENGTH = 500
MAX_REDIRECTS = 3
REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})
NO_GUIDELINE_RESULT = (
    "Không tìm thấy thông tin phù hợp trong các nguồn hướng dẫn y khoa thuộc "
    "allowlist. Vui lòng hỏi cụ thể hơn hoặc tham khảo chuyên gia."
)
SEARCH_UNAVAILABLE = (
    "Nguồn hướng dẫn y khoa hiện chưa được cấu hình hoặc tạm thời không khả "
    "dụng. Hệ thống sẽ không tạo câu trả lời không có nguồn."
)
SENSITIVE_SEARCH_REFUSAL = (
    "Không được gửi dữ liệu nhận dạng bệnh nhân tới dịch vụ tìm kiếm bên "
    "ngoài. Vui lòng dùng chức năng tra cứu bệnh án cho câu hỏi này."
)
RATE_LIMIT_RESPONSE = (
    "Bạn đã đạt giới hạn tra cứu hướng dẫn y khoa trong một phút. Vui lòng "
    "thử lại sau."
)
BUDGET_EXHAUSTED_RESPONSE = (
    "Ngân sách tra cứu hướng dẫn y khoa hôm nay đã hết. Hệ thống sẽ không "
    "tạo câu trả lời không có nguồn."
)
CIRCUIT_OPEN_RESPONSE = (
    "Nguồn hướng dẫn y khoa đang tạm ngưng sau nhiều lỗi liên tiếp. Vui lòng "
    "thử lại sau."
)
TOOL_CHAIN_WARNING = (
    "Các nguồn được hiển thị riêng theo thứ tự ưu tiên đã cấu hình; hệ thống "
    "không tự kết luận khi các hướng dẫn khác nhau."
)

# Broad domains are narrowed again with path rules after search returns.
APPROVED_SEARCH_DOMAINS = (
    "who.int",
    "nice.org.uk",
    "cdc.gov",
    "emohbackup.moh.gov.vn",
    "ydct.moh.gov.vn",
)
APPROVED_PATHS = {
    "who.int": (
        re.compile(r"^/publications/who-guidelines/?$", re.IGNORECASE),
        re.compile(r"^/publications/i/item/", re.IGNORECASE),
        re.compile(r"^/news-room/fact-sheets/detail/", re.IGNORECASE),
    ),
    "nice.org.uk": (
        re.compile(r"^/guidance(?:/|$)", re.IGNORECASE),
    ),
    "cdc.gov": (
        re.compile(r"^/[^/]+/hcp/clinical-guidance(?:/|$)", re.IGNORECASE),
    ),
    "emohbackup.moh.gov.vn": (
        re.compile(r"^/publish/attach/getfile/", re.IGNORECASE),
    ),
    "ydct.moh.gov.vn": (
        re.compile(r"^/static/files/uploads/", re.IGNORECASE),
    ),
}
SOURCE_PRIORITY = {
    "emohbackup.moh.gov.vn": (10, "Bộ Y tế Việt Nam"),
    "ydct.moh.gov.vn": (10, "Bộ Y tế Việt Nam"),
    "who.int": (20, "WHO"),
    "nice.org.uk": (30, "NICE"),
    "cdc.gov": (40, "CDC"),
}


class _NoRedirectHandler(HTTPRedirectHandler):
    """Expose redirects so every hop can be checked before following it."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        del req, fp, code, msg, headers, newurl
        return None


_SOURCE_OPENER = build_opener(_NoRedirectHandler())
_STATE_LOCK = Lock()
_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_RATE_EVENTS: Dict[str, Deque[float]] = defaultdict(deque)
_BUDGET_DAY = ""
_BUDGET_USED = 0
_CIRCUIT_FAILURES = 0
_CIRCUIT_OPEN_UNTIL = 0.0


class MedicalSearchBudgetExceeded(Exception):
    """Raised when a retry would exceed the local Tavily-call budget."""


SENSITIVE_PATTERNS = (
    re.compile(
        r"\b(?:patient[_\s-]?id|mã\s+bệnh\s+nhân|medical\s+record)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:room|phòng)\s*(?:number|số|[:#])?\s*\d+\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"\b(?:\+?\d[\s.-]?){8,15}\b"),
    re.compile(
        r"\b(?:bệnh\s+nhân|patient)\s+(?:tên\s+)?"
        r"[A-ZÀ-Ỹ][A-Za-zÀ-ỹ'-]+(?:\s+[A-ZÀ-Ỹ][A-Za-zÀ-ỹ'-]+){1,4}\b"
    ),
    re.compile(
        r"\b[A-Z][A-Za-z'-]+'s\s+"
        r"(?i:age|blood\s+pressure|diagnosis|medication|test\s+result)\b",
    ),
    re.compile(
        r"\b(?i:tuổi|huyết\s+áp|chẩn\s+đoán|thuốc|kết\s+quả)\s+của\s+"
        r"[A-ZÀ-Ỹ][A-Za-zÀ-ỹ'-]+\b",
    ),
)


def contains_sensitive_patient_data(question: str) -> bool:
    """Conservatively detect identifiers that must not leave the application."""
    if not isinstance(question, str) or not question.strip():
        return True
    if len(question) > MAX_QUESTION_LENGTH:
        return True
    return any(pattern.search(question) for pattern in SENSITIVE_PATTERNS)


def is_approved_source_url(url: str) -> bool:
    """Require HTTPS plus an exact approved host and approved path."""
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        if parsed.scheme.lower() != "https" or not host:
            return False
        if parsed.username or parsed.password or parsed.port not in (None, 443):
            return False
    except ValueError:
        return False

    path = unquote(parsed.path or "/")
    if ".." in path or "//" in path:
        return False

    for approved_host, patterns in APPROVED_PATHS.items():
        valid_hosts = {approved_host, f"www.{approved_host}"}
        if host in valid_hosts:
            return any(pattern.search(path) for pattern in patterns)
    return False


def resolve_approved_final_url(
    url: str,
    opener: Optional[Callable[..., Any]] = None,
    resolver: Optional[Callable[..., Any]] = None,
    max_redirects: int = MAX_REDIRECTS,
) -> Optional[str]:
    """Follow a bounded redirect chain only across approved, public hosts."""
    current_url = url
    open_source = opener or _open_source_without_redirect
    resolve_host = resolver or socket.getaddrinfo

    for redirect_count in range(max_redirects + 1):
        if not is_approved_source_url(current_url):
            return None
        host = urlparse(current_url).hostname or ""
        if not _host_has_only_public_addresses(host, resolve_host):
            return None

        request = Request(
            current_url,
            headers={"User-Agent": "HealthcareGraphRAG/1.0"},
            method="HEAD",
        )
        try:
            with open_source(request, timeout=5) as response:
                status = int(getattr(response, "status", 200))
                headers = getattr(response, "headers", {})
                response_url = str(
                    response.geturl() if hasattr(response, "geturl")
                    else current_url
                )
        except HTTPError as error:
            status = error.code
            headers = error.headers or {}
            response_url = current_url
        except (URLError, TimeoutError, ValueError, OSError):
            return None

        if status in REDIRECT_CODES:
            if redirect_count >= max_redirects:
                return None
            location = headers.get("Location")
            if not location:
                return None
            current_url = urljoin(current_url, location)
            continue
        if 200 <= status < 300 and is_approved_source_url(response_url):
            return response_url
        return None
    return None


def _open_source_without_redirect(request: Request, timeout: int):
    return _SOURCE_OPENER.open(request, timeout=timeout)


def _host_has_only_public_addresses(
    host: str, resolver: Callable[..., Any]
) -> bool:
    """Reject unresolved, private, loopback, link-local and reserved targets."""
    try:
        records = resolver(host, 443, type=socket.SOCK_STREAM)
        addresses = {record[4][0] for record in records}
        return bool(addresses) and all(
            ipaddress.ip_address(address).is_global for address in addresses
        )
    except (OSError, ValueError, TypeError):
        return False


def _source_metadata(url: str) -> Tuple[int, str]:
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    return SOURCE_PRIORITY.get(host, (999, "Nguồn khác"))


def _cache_key(question: str, max_results: int, min_score: float) -> str:
    normalized = " ".join(question.casefold().split())
    material = f"{normalized}|{max_results}|{min_score}|v1"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _get_cached_result(key: str, now: float) -> Optional[Dict[str, Any]]:
    with _STATE_LOCK:
        cached = _CACHE.get(key)
        if not cached:
            return None
        expires_at, result = cached
        if expires_at <= now:
            _CACHE.pop(key, None)
            return None
        return copy.deepcopy(result)


def _store_cached_result(
    key: str, result: Dict[str, Any], now: float, ttl_seconds: int
) -> None:
    if ttl_seconds <= 0:
        return
    with _STATE_LOCK:
        _CACHE[key] = (now + ttl_seconds, copy.deepcopy(result))


def _claim_provider_call(
    actor_id: str,
    now: float,
    utc_day: str,
    rate_limit_per_minute: int,
    daily_budget: int,
    circuit_failure_threshold: int,
) -> Optional[str]:
    """Atomically apply cache-independent operational limits."""
    global _BUDGET_DAY, _BUDGET_USED  # pylint: disable=global-statement
    with _STATE_LOCK:
        if _CIRCUIT_OPEN_UNTIL > now:
            return "circuit_open"

        actor_key = hashlib.sha256(actor_id.encode("utf-8")).hexdigest()
        events = _RATE_EVENTS[actor_key]
        while events and events[0] <= now - 60:
            events.popleft()
        if rate_limit_per_minute <= 0 or len(events) >= rate_limit_per_minute:
            return "rate_limited"

        if _BUDGET_DAY != utc_day:
            _BUDGET_DAY = utc_day
            _BUDGET_USED = 0
        if daily_budget <= 0 or _BUDGET_USED >= daily_budget:
            return "budget_exhausted"

        if circuit_failure_threshold <= 0:
            return "circuit_open"
        events.append(now)
        _BUDGET_USED += 1
        return None


def _record_provider_success() -> None:
    global _CIRCUIT_FAILURES, _CIRCUIT_OPEN_UNTIL  # pylint: disable=global-statement
    with _STATE_LOCK:
        _CIRCUIT_FAILURES = 0
        _CIRCUIT_OPEN_UNTIL = 0.0


def _claim_retry_budget(utc_day: str, daily_budget: int) -> bool:
    global _BUDGET_DAY, _BUDGET_USED  # pylint: disable=global-statement
    with _STATE_LOCK:
        if _BUDGET_DAY != utc_day:
            _BUDGET_DAY = utc_day
            _BUDGET_USED = 0
        if daily_budget <= 0 or _BUDGET_USED >= daily_budget:
            return False
        _BUDGET_USED += 1
        return True


def _record_provider_failure(
    now: float, threshold: int, cooldown_seconds: int
) -> None:
    global _CIRCUIT_FAILURES, _CIRCUIT_OPEN_UNTIL  # pylint: disable=global-statement
    with _STATE_LOCK:
        _CIRCUIT_FAILURES += 1
        if threshold > 0 and _CIRCUIT_FAILURES >= threshold:
            _CIRCUIT_OPEN_UNTIL = now + max(1, cooldown_seconds)


def _reset_runtime_state_for_tests() -> None:
    """Reset process-local controls; intended only for isolated unit tests."""
    global _BUDGET_DAY, _BUDGET_USED  # pylint: disable=global-statement
    global _CIRCUIT_FAILURES, _CIRCUIT_OPEN_UNTIL  # pylint: disable=global-statement
    with _STATE_LOCK:
        _CACHE.clear()
        _RATE_EVENTS.clear()
        _BUDGET_DAY = ""
        _BUDGET_USED = 0
        _CIRCUIT_FAILURES = 0
        _CIRCUIT_OPEN_UNTIL = 0.0


def search_medical_guidelines(
    question: str,
    api_key: str,
    max_results: int = 3,
    min_score: float = 0.5,
    opener: Optional[Callable[..., Any]] = None,
    source_opener: Optional[Callable[..., Any]] = None,
    resolver: Optional[Callable[..., Any]] = None,
    actor_id: str = "anonymous",
    cache_ttl_seconds: int = 300,
    rate_limit_per_minute: int = 10,
    daily_budget: int = 1000,
    max_retries: int = 1,
    max_retry_delay_seconds: float = 2.0,
    circuit_failure_threshold: int = 3,
    circuit_cooldown_seconds: int = 60,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    utcnow: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> Dict[str, Any]:
    """Retrieve allowlisted source snippets for discovery without synthesis."""
    if contains_sensitive_patient_data(question):
        audit_event("medical_search_rejected_sensitive_input", level="warning")
        return {"status": "privacy_denied", "response": SENSITIVE_SEARCH_REFUSAL}
    if not api_key:
        audit_event("medical_search_unavailable", reason="missing_api_key")
        return {"status": "unavailable", "response": SEARCH_UNAVAILABLE}

    safe_max_results = max(1, min(int(max_results), 5))
    now = clock()
    key = _cache_key(question, safe_max_results, min_score)
    cached = _get_cached_result(key, now)
    if cached is not None:
        audit_event("medical_search_cache_hit")
        cached["cache_hit"] = True
        return cached

    utc_day = utcnow().date().isoformat()
    limit_reason = _claim_provider_call(
        actor_id=actor_id or "anonymous",
        now=now,
        utc_day=utc_day,
        rate_limit_per_minute=rate_limit_per_minute,
        daily_budget=daily_budget,
        circuit_failure_threshold=circuit_failure_threshold,
    )
    if limit_reason:
        responses = {
            "rate_limited": RATE_LIMIT_RESPONSE,
            "budget_exhausted": BUDGET_EXHAUSTED_RESPONSE,
            "circuit_open": CIRCUIT_OPEN_RESPONSE,
        }
        audit_event("medical_search_blocked", reason=limit_reason)
        return {"status": limit_reason, "response": responses[limit_reason]}

    payload = {
        "query": question.strip(),
        "search_depth": "basic",
        "include_domains": list(APPROVED_SEARCH_DOMAINS),
        "max_results": safe_max_results,
        "include_answer": False,
        "include_raw_content": False,
    }
    request = Request(
        TAVILY_SEARCH_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        body = _request_search_json(
            request=request,
            opener=opener or urlopen,
            max_retries=max(0, min(int(max_retries), 2)),
            max_retry_delay_seconds=max(0.0, max_retry_delay_seconds),
            sleeper=sleeper,
            before_retry=lambda: _claim_retry_budget(utc_day, daily_budget),
        )
        _record_provider_success()
    except MedicalSearchBudgetExceeded:
        audit_event("medical_search_blocked", reason="budget_exhausted_on_retry")
        return {
            "status": "budget_exhausted",
            "response": BUDGET_EXHAUSTED_RESPONSE,
        }
    except (HTTPError, URLError, TimeoutError, ValueError, OSError) as error:
        logger.warning("Medical guideline search failed: %s", type(error).__name__)
        _record_provider_failure(
            now=clock(),
            threshold=circuit_failure_threshold,
            cooldown_seconds=circuit_cooldown_seconds,
        )
        audit_event("medical_search_unavailable", reason=type(error).__name__)
        return {"status": "unavailable", "response": SEARCH_UNAVAILABLE}

    approved_results: List[Dict[str, Any]] = []
    seen_urls = set()
    for result in body.get("results", []) if isinstance(body, dict) else []:
        if not isinstance(result, dict):
            continue
        url = str(result.get("url") or "").strip()
        try:
            score = float(result.get("score", 0))
        except (TypeError, ValueError):
            continue
        if score < min_score or url in seen_urls or not is_approved_source_url(url):
            continue
        final_url = resolve_approved_final_url(
            url,
            opener=source_opener,
            resolver=resolver,
        )
        if not final_url:
            continue
        title = _clean_text(result.get("title"), 180)
        content = _clean_text(result.get("content"), 700)
        if not title or not content:
            continue
        seen_urls.add(url)
        priority, source_name = _source_metadata(final_url)
        publication_date = _clean_text(
            result.get("published_date") or result.get("publishedDate"), 40
        ) or None
        approved_results.append({
            "title": title,
            "url": final_url,
            "provider_url": url,
            "final_url": final_url,
            "content": content,
            "content_hash": hashlib.sha256(
                content.encode("utf-8")
            ).hexdigest(),
            "score": score,
            "source_name": source_name,
            "source_priority": priority,
            "publication_date": publication_date,
            "document_version": None,
        })

    approved_results.sort(
        key=lambda item: (item["source_priority"], -item["score"], item["url"])
    )
    approved_results = approved_results[:safe_max_results]

    if not approved_results:
        audit_event("medical_search_no_approved_results")
        result = {"status": "not_found", "response": NO_GUIDELINE_RESULT}
        _store_cached_result(key, result, clock(), cache_ttl_seconds)
        return result

    retrieved_at = utcnow().astimezone(timezone.utc).isoformat()
    lines = ["Thông tin discovery từ các nguồn y khoa trong allowlist:"]
    evidence = []
    for index, result in enumerate(approved_results, start=1):
        source_id = f"S{index}"
        lines.extend([
            "",
            f'[{source_id}] {result["source_name"]}: {result["title"]}',
            result["content"],
            f'Nguồn: {result["url"]}',
        ])
        evidence.append({
            "id": source_id,
            "retrieved_at": retrieved_at,
            **result,
        })
    lines.extend([
        "",
        TOOL_CHAIN_WARNING,
        "Đây là thông tin tham khảo từ tài liệu nguồn, không phải chẩn đoán "
        "hoặc chỉ định điều trị cho một bệnh nhân cụ thể.",
    ])
    audit_event("medical_search_completed", source_count=len(evidence))
    rendered = {
        "status": "success",
        "response": "\n".join(lines),
        "evidence": evidence,
        "retrieved_at": retrieved_at,
        "cache_hit": False,
    }
    _store_cached_result(key, rendered, clock(), cache_ttl_seconds)
    return rendered


def _request_search_json(
    request: Request,
    opener: Callable[..., Any],
    max_retries: int,
    max_retry_delay_seconds: float,
    sleeper: Callable[[float], None],
    before_retry: Callable[[], bool],
) -> Dict[str, Any]:
    """Call Tavily with bounded retry for 429 and transient server errors."""
    for attempt in range(max_retries + 1):
        try:
            with opener(request, timeout=10) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise ValueError("Medical search response exceeded size limit")
            body = json.loads(raw.decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("Medical search response must be an object")
            return body
        except HTTPError as error:
            retryable = error.code == 429 or 500 <= error.code < 600
            if not retryable or attempt >= max_retries:
                raise
            delay = _retry_delay(error, attempt)
            if delay > max_retry_delay_seconds:
                raise
            if not before_retry():
                raise MedicalSearchBudgetExceeded from error
            sleeper(delay)
        except (URLError, TimeoutError, OSError):
            if attempt >= max_retries:
                raise
            if not before_retry():
                raise MedicalSearchBudgetExceeded
            sleeper(min(max_retry_delay_seconds, 0.25 * (2 ** attempt)))
    raise ValueError("Medical search retry loop ended unexpectedly")


def _retry_delay(error: HTTPError, attempt: int) -> float:
    value = (error.headers or {}).get("Retry-After")
    if not value:
        return 0.25 * (2 ** attempt)
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(
                0.0,
                (retry_at - datetime.now(timezone.utc)).total_seconds(),
            )
        except (TypeError, ValueError, OverflowError):
            return float("inf")


def _clean_text(value: Any, limit: int) -> str:
    """Collapse whitespace/control characters and cap untrusted provider text."""
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit].rstrip()
