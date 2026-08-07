"""Allowlisted, retrieval-only medical guideline search.

The search provider is used only to retrieve snippets. The application disables
provider-generated answers, post-filters every result URL, and returns the
approved snippets verbatim with citations. Patient-identifying queries are
rejected before any network request.
"""
import json
import re
from typing import Any, Callable, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from src.handlers.security_guardrails import audit_event
from src.helpers.logging_config import logger


TAVILY_SEARCH_URL = "https://api.tavily.com/search"
MAX_RESPONSE_BYTES = 1_000_000
MAX_QUESTION_LENGTH = 500
NO_GUIDELINE_RESULT = (
    "Không tìm thấy thông tin phù hợp trong các nguồn hướng dẫn y khoa đã "
    "được phê duyệt. Vui lòng hỏi cụ thể hơn hoặc tham khảo chuyên gia."
)
SEARCH_UNAVAILABLE = (
    "Nguồn hướng dẫn y khoa hiện chưa được cấu hình hoặc tạm thời không khả "
    "dụng. Hệ thống sẽ không tạo câu trả lời không có nguồn."
)
SENSITIVE_SEARCH_REFUSAL = (
    "Không được gửi dữ liệu nhận dạng bệnh nhân tới dịch vụ tìm kiếm bên "
    "ngoài. Vui lòng dùng chức năng tra cứu bệnh án cho câu hỏi này."
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
        r"\b[A-ZÀ-Ỹ][A-Za-zÀ-ỹ'-]+(?:'s)?\s+"
        r"(?i:age|blood\s+pressure|diagnosis|medication|test\s+result|"
        r"bao\s+nhiêu\s+tuổi|huyết\s+áp|chẩn\s+đoán|thuốc|kết\s+quả)\b",
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


def search_medical_guidelines(
    question: str,
    api_key: str,
    max_results: int = 3,
    min_score: float = 0.5,
    opener: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Retrieve approved source snippets and render citations without synthesis."""
    if contains_sensitive_patient_data(question):
        audit_event("medical_search_rejected_sensitive_input", level="warning")
        return {"status": "privacy_denied", "response": SENSITIVE_SEARCH_REFUSAL}
    if not api_key:
        audit_event("medical_search_unavailable", reason="missing_api_key")
        return {"status": "unavailable", "response": SEARCH_UNAVAILABLE}

    safe_max_results = max(1, min(int(max_results), 5))
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
        open_request = opener or urlopen
        with open_request(request, timeout=10) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("Medical search response exceeded size limit")
        body = json.loads(raw.decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
        logger.warning("Medical guideline search failed: %s", type(error).__name__)
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
        title = _clean_text(result.get("title"), 180)
        content = _clean_text(result.get("content"), 700)
        if not title or not content:
            continue
        seen_urls.add(url)
        approved_results.append({
            "title": title,
            "url": url,
            "content": content,
            "score": score,
        })
        if len(approved_results) >= safe_max_results:
            break

    if not approved_results:
        audit_event("medical_search_no_approved_results")
        return {"status": "not_found", "response": NO_GUIDELINE_RESULT}

    lines = ["Thông tin từ các nguồn hướng dẫn y khoa đã được phê duyệt:"]
    evidence = []
    for index, result in enumerate(approved_results, start=1):
        source_id = f"S{index}"
        lines.extend([
            "",
            f'[{source_id}] {result["title"]}',
            result["content"],
            f'Nguồn: {result["url"]}',
        ])
        evidence.append({"id": source_id, **result})
    lines.extend([
        "",
        "Đây là thông tin tham khảo từ tài liệu nguồn, không phải chẩn đoán "
        "hoặc chỉ định điều trị cho một bệnh nhân cụ thể.",
    ])
    audit_event("medical_search_completed", source_count=len(evidence))
    return {
        "status": "success",
        "response": "\n".join(lines),
        "evidence": evidence,
    }


def _clean_text(value: Any, limit: int) -> str:
    """Collapse whitespace/control characters and cap untrusted provider text."""
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit].rstrip()
