"""Deterministic security guardrails for scoped medical GraphRAG queries.

Authorization is enforced by rewriting a deliberately small, supported Cypher
subset and rejecting complex syntax fail-closed. Input checks are heuristics and
do not replace authorization. Result checks flag suspicious values but are not a
clinical diagnosis or a semantic proof of LLM-to-Cypher correctness.
"""
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.helpers.logging_config import logger


MAX_RESULT_ROWS = 20
MEDICAL_RANGES = {
    "age": (0, 120),
    "heart_rate": (20, 250),
    "temperature_celsius": (30, 45),
}
PROMPT_INJECTION_PATTERNS = (
    r"\bignore\s+(?:all\s+)?previous\s+instructions?\b",
    r"\b(?:reveal|show|print)\s+(?:the\s+)?system\s+prompt\b",
    r"\breturn\s+all\s+(?:records?|patients?|data)\b",
    r"\bbypass\s+(?:the\s+)?(?:filter|authorization|scope|security)\b",
    r"\bdisable\s+(?:the\s+)?(?:filter|authorization|scope|security)\b",
    r"\bignore\s+(?:the\s+)?(?:doctor|patient)\s+(?:filter|scope)\b",
    r"\bbỏ\s+qua\s+(?:mọi\s+)?(?:chỉ\s+thị|bộ\s+lọc|phân\s+quyền)\b",
    r"\btrả\s+về\s+(?:tất\s+cả|toàn\s+bộ)\s+(?:bản\s+ghi|bệnh\s+nhân|dữ\s+liệu)\b",
)
COMPLEX_CYPHER_PATTERN = re.compile(
    r"\b(?:UNION|CALL|WITH|UNWIND|USE|YIELD|LOAD\s+CSV|FOREACH)\b",
    re.IGNORECASE,
)


class AuthorizationScopeError(ValueError):
    """Raised when a generated query cannot be safely doctor-scoped."""


@dataclass
class ValidationResult:
    """Fail-closed result validation outcome plus non-blocking review flags."""

    valid: bool
    rows: List[Dict[str, Any]]
    reason: Optional[str] = None
    user_message: Optional[str] = None
    flagged_for_review: bool = False
    possible_semantic_mismatch: bool = False
    missing_fields: List[Tuple[int, str]] = field(default_factory=list)
    field_warnings: Dict[Tuple[int, str], str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def audit_event(event: str, level: str = "info", **details: Any) -> None:
    """Write one JSON audit record; production should route this to a secure sink."""
    payload = {"event": event, **details}
    log_method = getattr(logger, level, logger.info)
    log_method("audit=%s", json.dumps(payload, ensure_ascii=False, default=str))


def enforce_scope(cypher: str, doctor_id: str) -> str:
    """Inject mandatory doctor scope into simple Cypher or reject fail-closed.

    UNION, subqueries, WITH pipelines and other syntax that this small rewriter
    cannot prove safe are rejected instead of being handled with brittle regex.
    The returned query always uses the ``$doctor_id`` Neo4j parameter.
    """
    original = cypher
    if not isinstance(doctor_id, str) or not doctor_id.strip():
        _reject_scope(doctor_id, original, "missing_doctor_id")
    if not isinstance(cypher, str) or not cypher.strip():
        _reject_scope(doctor_id, original, "empty_cypher")

    masked = _mask_string_literals(cypher)
    if (
        COMPLEX_CYPHER_PATTERN.search(masked)
        or "{" in masked
        or "}" in masked
        or "//" in masked
        or "/*" in masked
        or "*/" in masked
        or "`" in masked
        or ";" in masked
    ):
        _reject_scope(doctor_id, original, "unsupported_complex_cypher")

    return_matches = list(re.finditer(r"\bRETURN\b", masked, re.IGNORECASE))
    if len(return_matches) != 1:
        _reject_scope(doctor_id, original, "query_must_have_one_return")

    patient_labels = re.findall(r":\s*Patient\b", masked, re.IGNORECASE)
    patient_variables = list(dict.fromkeys(re.findall(
        r"\(\s*([A-Za-z_]\w*)\s*:\s*Patient\b", masked, re.IGNORECASE
    )))
    if not patient_variables or len(patient_labels) != len(patient_variables):
        _reject_scope(doctor_id, original, "patient_anchor_is_missing_or_ambiguous")

    unscoped_variables = []
    all_scope_mentions = re.findall(
        r"\b([A-Za-z_]\w*)\.attending_doctor_id\b", masked, re.IGNORECASE
    )
    if any(
        variable.casefold() not in {item.casefold() for item in patient_variables}
        for variable in all_scope_mentions
    ):
        _reject_scope(doctor_id, original, "scope_filter_uses_non_patient_variable")

    for variable in patient_variables:
        property_pattern = re.compile(
            rf"\b{re.escape(variable)}\.attending_doctor_id\b", re.IGNORECASE
        )
        correct_pattern = re.compile(
            rf"(?:\b{re.escape(variable)}\.attending_doctor_id\s*=\s*\$doctor_id\b"
            rf"|\$doctor_id\s*=\s*\b{re.escape(variable)}\.attending_doctor_id\b)",
            re.IGNORECASE,
        )
        if property_pattern.search(masked):
            without_correct_scope = correct_pattern.sub("__DOCTOR_SCOPE__", masked)
            if property_pattern.search(without_correct_scope):
                _reject_scope(doctor_id, original, "conflicting_doctor_scope")
            if re.search(r"\b(?:OR|XOR|NOT)\b", masked, re.IGNORECASE):
                _reject_scope(doctor_id, original, "unsafe_scope_boolean_logic")
            if re.search(
                r"(?:__DOCTOR_SCOPE__\s*(?:=|<>|<=|>=|<|>|IS)"
                r"|(?:=|<>|<=|>=|<|>|IS)\s*__DOCTOR_SCOPE__)",
                without_correct_scope,
                re.IGNORECASE,
            ):
                _reject_scope(doctor_id, original, "unsafe_scope_expression")
        else:
            unscoped_variables.append(variable)

    if not unscoped_variables:
        audit_event(
            "authorization_scope_preserved",
            doctor_id=doctor_id,
            cypher=original,
        )
        return cypher

    scope_expression = " AND ".join(
        f"{variable}.attending_doctor_id = $doctor_id"
        for variable in unscoped_variables
    )
    return_index = return_matches[0].start()
    scoped_query = (
        f"{cypher[:return_index].rstrip()} "
        f"WITH * WHERE {scope_expression} "
        f"{cypher[return_index:].lstrip()}"
    )
    audit_event(
        "authorization_scope_injected",
        doctor_id=doctor_id,
        cypher=original,
        scoped_cypher=scoped_query,
    )
    return scoped_query


def check_prompt_injection(user_question: str) -> bool:
    """Return True when a deterministic pattern marks the input as suspicious."""
    question = user_question or ""
    suspicious = any(
        re.search(pattern, question, re.IGNORECASE)
        for pattern in PROMPT_INJECTION_PATTERNS
    )
    if suspicious:
        audit_event(
            "prompt_injection_suspected",
            level="warning",
            user_question=question,
        )
    else:
        audit_event("prompt_injection_check_passed")
    return suspicious


def check_input_scope(
    user_question: str,
    doctor_id: str,
    scope_lookup: Optional[Callable[[str, str, str], bool]] = None,
) -> bool:
    """Early-check an explicit patient reference; unknown inputs fail closed.

    This is only a cost-saving optimization. ``enforce_scope`` remains mandatory
    even when this function returns True.
    """
    reference = extract_patient_reference(user_question)
    if reference is None:
        audit_event(
            "input_scope_check_skipped",
            doctor_id=doctor_id,
            reason="no_explicit_patient_reference",
        )
        return True
    if scope_lookup is None:
        audit_event(
            "input_scope_review_required",
            level="warning",
            doctor_id=doctor_id,
            reason="scope_lookup_unavailable",
            patient_reference=reference,
        )
        return False

    field_name, value = reference
    try:
        allowed = bool(scope_lookup(field_name, value, doctor_id))
    except Exception as error:  # Database uncertainty must not become allow.
        audit_event(
            "input_scope_review_required",
            level="warning",
            doctor_id=doctor_id,
            reason="scope_lookup_failed",
            patient_reference=reference,
            error=str(error),
        )
        return False

    if not allowed:
        audit_event(
            "input_scope_rejected",
            level="warning",
            doctor_id=doctor_id,
            patient_reference=reference,
        )
    else:
        audit_event(
            "input_scope_check_passed",
            doctor_id=doctor_id,
            patient_reference=reference,
        )
    return allowed


def extract_patient_reference(user_question: str) -> Optional[Tuple[str, str]]:
    """Extract only explicit patient IDs, quoted names or clear title-case names."""
    question = user_question or ""
    patient_id_match = re.search(
        r"(?:patient[_\s-]?id|mã\s+bệnh\s+nhân)\s*[:=#-]?\s*([A-Za-z0-9_-]+)",
        question,
        re.IGNORECASE,
    )
    if patient_id_match:
        return "patient_id", patient_id_match.group(1)

    quoted_name_match = re.search(
        r"(?:bệnh\s+nhân|patient)\s+(?:tên\s+)?[\"']([^\"']+)[\"']",
        question,
        re.IGNORECASE,
    )
    if quoted_name_match:
        return "name", quoted_name_match.group(1).strip()

    titled_name_match = re.search(
        r"(?:bệnh\s+nhân|patient)\s+(?:tên\s+)?"
        r"([A-ZÀ-Ỹ][A-Za-zÀ-ỹ'-]*(?:\s+[A-ZÀ-Ỹ][A-Za-zÀ-ỹ'-]*){0,3})",
        question,
    )
    if titled_name_match:
        return "name", titled_name_match.group(1).strip()
    return None


def validate_result(
    rows: List[Dict[str, Any]],
    cypher: str,
    user_question: str,
    max_rows: int = MAX_RESULT_ROWS,
    medical_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> ValidationResult:
    """Validate result shape/count and attach non-blocking range/intent flags."""
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        return ValidationResult(
            valid=False,
            rows=[],
            reason="invalid_result_shape",
            user_message="Kết quả dữ liệu không đúng định dạng an toàn.",
        )
    if len(rows) > max_rows:
        audit_event(
            "result_row_limit_exceeded",
            level="warning",
            row_count=len(rows),
            max_rows=max_rows,
            cypher=cypher,
        )
        return ValidationResult(
            valid=False,
            rows=rows,
            reason="row_limit_exceeded",
            user_message=(
                "Kết quả trả về nhiều hơn dự kiến, vui lòng hỏi cụ thể hơn."
            ),
        )

    ranges = medical_ranges or MEDICAL_RANGES
    result = ValidationResult(valid=True, rows=rows)
    for row_index, row in enumerate(rows):
        for field_name, value in row.items():
            if value is None:
                result.missing_fields.append((row_index, field_name))
                continue
            range_name = _matching_range(field_name, ranges)
            if (
                range_name is None
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                continue
            minimum, maximum = ranges[range_name]
            if value < minimum or value > maximum:
                warning = "giá trị bất thường, vui lòng xác minh lại"
                result.field_warnings[(row_index, field_name)] = warning
                result.flagged_for_review = True
                result.warnings.append(
                    f"{field_name}={value} nằm ngoài khoảng {minimum}-{maximum}"
                )

    question = (user_question or "").casefold()
    asks_latest = any(
        keyword in question
        for keyword in ("gần nhất", "mới nhất", "latest", "most recent")
    )
    has_latest_query_shape = bool(
        re.search(r"\bORDER\s+BY\b[\s\S]*\bDESC\b", cypher, re.IGNORECASE)
        and re.search(r"\bLIMIT\s+\d+\b", cypher, re.IGNORECASE)
    )
    if asks_latest and not has_latest_query_shape:
        result.possible_semantic_mismatch = True
        result.flagged_for_review = True
        result.warnings.append(
            "Câu hỏi yêu cầu dữ liệu mới nhất nhưng Cypher thiếu ORDER BY DESC LIMIT."
        )

    if result.flagged_for_review:
        audit_event(
            "result_flagged_for_review",
            level="warning",
            cypher=cypher,
            warnings=result.warnings,
        )
    else:
        audit_event(
            "result_validation_passed",
            row_count=len(rows),
            cypher=cypher,
        )
    return result


def _reject_scope(doctor_id: str, cypher: str, reason: str) -> None:
    audit_event(
        "authorization_scope_rejected",
        level="warning",
        doctor_id=doctor_id,
        cypher=cypher,
        reason=reason,
    )
    raise AuthorizationScopeError(reason)


def _mask_string_literals(cypher: str) -> str:
    return re.sub(
        r"'(?:\\.|''|[^'])*'|\"(?:\\.|\"\"|[^\"])*\"",
        lambda match: " " * len(match.group(0)),
        cypher or "",
    )


def _matching_range(
    field_name: str, ranges: Dict[str, Tuple[float, float]]
) -> Optional[str]:
    normalized = field_name.casefold()
    for range_name in ranges:
        if normalized == range_name or normalized.endswith(f"_{range_name}"):
            return range_name
    return None
