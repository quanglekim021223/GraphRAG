"""Fail-closed rendering with optional, data-free natural templates.

The LLM may choose wording in the same call that creates Cypher, but only Neo4j
rows supply values. Invalid templates and per-row formatting errors fall back to
the less natural deterministic renderer; this trades flexibility for auditability.
Alias and character checks cannot prove that static prose is clinically neutral;
production medical interpretations still require approved phrases or rules.
"""
import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple, TypedDict


OUT_OF_SCOPE_RESPONSE = (
    "Hệ thống này chỉ hỗ trợ tra cứu dữ liệu bệnh án đã được phân quyền. "
    "Với kiến thức hoặc tư vấn y khoa tổng quát, vui lòng tham khảo nguồn "
    "chuyên môn đã được đơn vị của bạn phê duyệt."
)
ABSTAIN_RESPONSE = (
    "Tôi không thể biểu diễn an toàn dữ liệu truy xuất được, "
    "nên sẽ không đưa ra kết luận."
)
SCALAR_TYPES = (str, int, float, bool, type(None))
PLACEHOLDER_PATTERN = re.compile(r"\{(\w+)\}")
CONDITIONAL_PATTERN = re.compile(
    r"\b(?:if|else|otherwise|nếu|ngược\s+lại)\b", re.IGNORECASE
)
ALLOWED_TEMPLATE_PUNCTUATION = frozenset(".,;:!?-'\"()")


class EvidenceSource(TypedDict):
    """Exact database result cell used by one rendered sentence."""

    row: int
    field: str
    value: Any


class Evidence(TypedDict):
    """A rendered sentence and the result cells that support it."""

    id: str
    claim: str
    sources: List[EvidenceSource]


def validate_template(template: str, cypher: str) -> bool:
    """Accept only data-free placeholders backed by explicit RETURN aliases."""
    if not isinstance(template, str) or not template.strip():
        return False

    placeholders = PLACEHOLDER_PATTERN.findall(template)
    aliases = _extract_return_aliases(cypher)
    if not placeholders or not set(placeholders).issubset(aliases):
        return False

    static_text = PLACEHOLDER_PATTERN.sub("", template)
    if "{" in static_text or "}" in static_text:
        return False
    if re.search(r"\d", static_text):
        return False
    if CONDITIONAL_PATTERN.search(static_text):
        return False
    if any(
        not (
            character.isalpha()
            or character.isspace()
            or character in ALLOWED_TEMPLATE_PUNCTUATION
        )
        for character in static_text
    ):
        return False

    # Reject names or other filter values copied from the question into Cypher.
    normalized_template = template.casefold()
    if any(
        literal and literal.casefold() in normalized_template
        for literal in _extract_cypher_string_literals(cypher)
    ):
        return False
    return True


def build_grounded_output(
    query_result: List[Dict[str, Any]],
    response_template: Optional[str] = None,
    field_warnings: Optional[Dict[Tuple[int, str], str]] = None,
) -> Dict[str, Any]:
    """Build the existing response contract with optional safe natural wording."""
    answer, evidence, omitted_fields = _render_answer_details(
        query_result, response_template, field_warnings
    )

    if not answer:
        return _abstain("unsupported_result_shape", omitted_fields)

    if omitted_fields:
        answer += (
            "\n\nLưu ý: "
            f"{omitted_fields} trường đã được ẩn vì không phải giá trị vô hướng "
            "có thể kiểm chứng trực tiếp."
        )

    return {
        "grounded": True,
        "answer": answer,
        "evidence": evidence,
        "omitted_fields": omitted_fields,
        "reason": None,
    }


def render_answer(
    rows: List[Dict[str, Any]],
    template: str,
    field_warnings: Optional[Dict[Tuple[int, str], str]] = None,
) -> Tuple[str, List[Evidence]]:
    """Render each row; formatting failures fall back for that row only."""
    answer, evidence, _ = _render_answer_details(
        rows, template, field_warnings
    )
    return (answer or ABSTAIN_RESPONSE, evidence)


def _render_answer_details(
    rows: List[Dict[str, Any]],
    template: Optional[str],
    field_warnings: Optional[Dict[Tuple[int, str], str]] = None,
) -> Tuple[str, List[Evidence], int]:
    evidence: List[Evidence] = []
    answer_lines = []
    omitted_fields = 0
    template_fields = list(dict.fromkeys(
        PLACEHOLDER_PATTERN.findall(template or "")
    ))
    warnings = field_warnings or {}

    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            omitted_fields += 1
            continue

        omitted_fields += sum(
            1 for field, value in row.items()
            if not isinstance(field, str) or not _is_supported_scalar(value)
        )
        sentence = ""
        sources: List[EvidenceSource] = []

        if template_fields:
            try:
                if any(
                    field not in row or not _is_supported_scalar(row[field])
                    for field in template_fields
                ):
                    raise KeyError("Template field is missing or non-scalar")
                format_row = dict(row)
                for field in template_fields:
                    value = row[field]
                    if value is None:
                        format_row[field] = _missing_value_text(field)
                    elif (row_index, field) in warnings:
                        format_row[field] = (
                            f"{value} ({warnings[(row_index, field)]})"
                        )
                sentence = template.format(**format_row).strip()  # type: ignore[union-attr]
                if not sentence:
                    raise ValueError("Rendered template is empty")
                sources = [
                    {"row": row_index, "field": field, "value": row[field]}
                    for field in template_fields
                    if row[field] is not None
                ]
            except Exception:  # Fail closed for every formatting failure.
                sentence, sources = _render_deterministic_row(
                    row, row_index, warnings
                )
        else:
            sentence, sources = _render_deterministic_row(
                row, row_index, warnings
            )

        if not sentence:
            continue

        punctuation = "" if sentence.endswith((".", "!", "?")) else "."
        if sources:
            evidence_id = f"E{len(evidence) + 1}"
            evidence.append(
                {
                    "id": evidence_id,
                    "claim": sentence,
                    "sources": sources,
                }
            )
            answer_lines.append(f"- {sentence}{punctuation} [{evidence_id}]")
        else:
            answer_lines.append(f"- {sentence}{punctuation}")

    return "\n".join(answer_lines), evidence, omitted_fields


def _render_deterministic_row(
    row: Dict[str, Any],
    row_index: int,
    field_warnings: Optional[Dict[Tuple[int, str], str]] = None,
) -> Tuple[str, List[EvidenceSource]]:
    sources = []
    facts = []
    warnings = field_warnings or {}
    for field, value in row.items():
        if not isinstance(field, str) or not _is_supported_scalar(value):
            continue
        if value is None:
            facts.append(_missing_value_text(field))
            continue
        sources.append({"row": row_index, "field": field, "value": value})
        displayed_value = _display_value(value)
        if (row_index, field) in warnings:
            displayed_value += f" ({warnings[(row_index, field)]})"
        facts.append(f"{_humanize_field(field)}: {displayed_value}")
    return "; ".join(facts), sources


def _missing_value_text(field: str) -> str:
    return f"(không có dữ liệu {_humanize_field(field).lower()})"


def format_grounded_response(result: Dict[str, Any]) -> str:
    """Render evidence details for the outer ReAct agent."""
    response = str(result.get("response") or "No information found")
    evidence = result.get("evidence") or []
    if not evidence:
        return response

    evidence_lines = []
    for item in evidence:
        sources = ", ".join(
            f'row {source["row"]}.{source["field"]}='
            f'{json.dumps(source["value"], ensure_ascii=False, default=str)}'
            for source in item.get("sources", [])
        )
        evidence_lines.append(f'- [{item["id"]}] {sources}')

    return f'{response}\n\nEvidence:\n' + "\n".join(evidence_lines)


def select_controlled_agent_response(full_response: Dict[str, Any]) -> str:
    """Return only controlled GraphRAG or curated guideline-tool output."""
    messages = full_response.get("messages", [])
    current_turn = messages

    for index in range(len(messages) - 1, -1, -1):
        if _message_value(messages[index], "type") == "human":
            current_turn = messages[index + 1:]
            break

    for message in reversed(current_turn):
        if _message_value(message, "name") == "rag_tool":
            return str(_message_value(message, "content") or "No information found")

    for message in reversed(current_turn):
        if _message_value(message, "name") == "medical_guideline_tool":
            return str(_message_value(message, "content") or OUT_OF_SCOPE_RESPONSE)

    # Fail closed if ReAct answers directly instead of using an approved tool.
    return OUT_OF_SCOPE_RESPONSE


def _is_supported_scalar(value: Any) -> bool:
    """Allow only values that can be rendered without interpretation."""
    if not isinstance(value, SCALAR_TYPES):
        return False
    return not isinstance(value, float) or math.isfinite(value)


def _humanize_field(field: str) -> str:
    """Turn a semantic Cypher alias into a deterministic display label."""
    return field.replace("_", " ").strip().capitalize()


def _display_value(value: Any) -> str:
    """Format a scalar without changing its value or inventing a unit."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _message_value(message: Any, key: str) -> Any:
    """Read LangChain message objects and lightweight test dictionaries."""
    if isinstance(message, dict):
        return message.get(key)
    return getattr(message, key, None)


def _extract_return_aliases(cypher: str) -> set:
    """Extract explicit aliases from the final RETURN clause."""
    query_without_literals = re.sub(
        r"'(?:\\.|''|[^'])*'|\"(?:\\.|\"\"|[^\"])*\"",
        lambda match: " " * len(match.group(0)),
        cypher or "",
    )
    matches = list(re.finditer(r"\bRETURN\b", query_without_literals, re.IGNORECASE))
    if not matches:
        return set()
    return_body = query_without_literals[matches[-1].end():]
    return_body = re.split(
        r"\b(?:ORDER\s+BY|SKIP|LIMIT)\b",
        return_body,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return {
        alias
        for alias in re.findall(
            r"\bAS\s+([A-Za-z_]\w*)\b", return_body, re.IGNORECASE
        )
    }


def _extract_cypher_string_literals(cypher: str) -> List[str]:
    """Return quoted filter values so they cannot be copied into a template."""
    values = []
    for match in re.finditer(
        r"'((?:\\.|''|[^'])*)'|\"((?:\\.|\"\"|[^\"])*)\"", cypher or ""
    ):
        value = match.group(1) if match.group(1) is not None else match.group(2)
        values.append(value.replace("''", "'").replace('""', '"'))
    return values


def _abstain(reason: str, omitted_fields: int) -> Dict[str, Any]:
    """Return one consistent fail-closed result."""
    return {
        "grounded": False,
        "answer": ABSTAIN_RESPONSE,
        "evidence": [],
        "omitted_fields": omitted_fields,
        "reason": reason,
    }
