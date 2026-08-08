"""Policy-driven patient-record plus curated-guideline workflow.

The outer agent selects one bounded clinical intent. Python policy then decides
which patient fields may be retrieved and copied into a de-identified guideline
query. Administrative fields and patient identifiers never cross that boundary.
Patient and guideline evidence stay separate; no LLM synthesizes a diagnosis.
"""
import re
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from src.handlers.grounding_verifier import format_grounded_response
from src.handlers.medical_guideline_search import contains_sensitive_patient_data
from src.handlers.security_guardrails import audit_event, extract_patient_reference


MAX_EXPLICIT_TERMS = 5
MAX_FACTS = 10
MAX_FACT_LENGTH = 100
SAFE_FACT_PATTERN = re.compile(
    r"^[A-Za-zÀ-ỹ0-9][A-Za-zÀ-ỹ0-9 .,'()/+\-]*$"
)


@dataclass(frozen=True)
class CompositePolicy:
    """Allowlisted patient facts and handoff behavior for one clinical intent."""

    patient_question: str
    patient_aliases: Tuple[str, ...]
    handoff_label: str
    requires_explicit_terms: bool = False


COMPOSITE_POLICIES: Mapping[str, CompositePolicy] = {
    "drug_interaction": CompositePolicy(
        patient_question=(
            "Liệt kê tên thuốc đang được ghi nhận cho {patient}; "
            "trả về tên thuốc với alias medication_name."
        ),
        patient_aliases=("medication_name", "medicine_name", "drug_name"),
        handoff_label=(
            "Drug interaction, contraindication, or medication safety guidance "
            "for concurrent use of"
        ),
        requires_explicit_terms=True,
    ),
    "disease_guideline": CompositePolicy(
        patient_question=(
            "Liệt kê bệnh hoặc chẩn đoán đang được ghi nhận cho {patient}; "
            "trả về tên bệnh với alias disease_name."
        ),
        patient_aliases=("disease_name", "medical_condition", "condition_name"),
        handoff_label="Clinical guideline for recorded condition",
    ),
    "blood_type_compatibility": CompositePolicy(
        patient_question=(
            "Cho biết nhóm máu đang được ghi nhận cho {patient}; "
            "trả về nhóm máu với alias blood_type."
        ),
        patient_aliases=("blood_type",),
        handoff_label="Blood transfusion compatibility guidance for blood type",
    ),
}

COMPOSITE_CLARIFICATION = (
    "Vui lòng nêu rõ một bệnh nhân và mục đích đối chiếu guideline. Với tương "
    "tác thuốc, hãy ghi rõ tên thuốc cần kiểm tra."
)
NO_ALLOWED_FACTS = (
    "Không tìm thấy dữ kiện lâm sàng phù hợp với mục đích đã chọn trong kết quả "
    "bệnh án. Vui lòng kiểm tra lại bệnh nhân hoặc hỏi cụ thể hơn."
)
COMPOSITE_SAFETY_NOTE = (
    "Các dữ kiện bệnh án và nội dung guideline được hiển thị riêng; hệ thống "
    "không tự kết luận quan hệ nhân quả, chẩn đoán hoặc thay đổi điều trị."
)


def run_patient_guideline_workflow(
    question: str,
    intent: str,
    explicit_terms: Sequence[str],
    doctor_id: str,
    graphrag: Any,
    guideline_retriever: Callable[..., Dict[str, Any]],
    guideline_options: Dict[str, Any],
) -> Dict[str, Any]:
    """Run one authorized lookup and one policy-filtered guideline retrieval."""
    policy = COMPOSITE_POLICIES.get(intent)
    if policy is None:
        return _clarification("unsupported_composite_intent")

    patient_reference = extract_patient_reference(question)
    if patient_reference is None:
        return _clarification("missing_patient_reference")

    reference_field, reference_value = patient_reference
    terms = _validate_explicit_terms(question, explicit_terms, reference_value)
    if terms is None or (policy.requires_explicit_terms and not terms):
        return _clarification("missing_or_unverified_explicit_terms")
    if not policy.requires_explicit_terms and terms:
        return _clarification("unexpected_explicit_terms")

    patient_question = policy.patient_question.format(
        patient=_patient_description(reference_field, reference_value)
    )
    patient_result = graphrag.run(patient_question, doctor_id)
    if patient_result.get("status") != "success":
        audit_event(
            "patient_guideline_patient_lookup_stopped",
            intent=intent,
            status=patient_result.get("status"),
            reason=patient_result.get("reason"),
        )
        return {
            "status": patient_result.get("status", "patient_lookup_failed"),
            "response": str(
                patient_result.get("response") or COMPOSITE_CLARIFICATION
            ),
            "patient_evidence": patient_result.get("evidence", []),
            "guideline_evidence": [],
        }

    patient_facts = _extract_allowed_facts(
        patient_result.get("result", []), policy.patient_aliases
    )
    if not patient_facts:
        audit_event(
            "patient_guideline_no_allowed_facts", level="warning", intent=intent
        )
        return {
            "status": "needs_clarification",
            "response": NO_ALLOWED_FACTS,
            "patient_evidence": patient_result.get("evidence", []),
            "guideline_evidence": [],
        }

    guideline_question = _build_guideline_question(
        intent, policy, terms, patient_facts
    )
    handoff_values = [*terms, *patient_facts]
    if (
        any(
            _contains_patient_reference(value, reference_value)
            for value in handoff_values
        )
        or _contains_explicit_phrase(
            guideline_question.casefold(), reference_value.casefold()
        )
        or contains_sensitive_patient_data(guideline_question)
    ):
        audit_event(
            "patient_guideline_handoff_rejected",
            level="warning",
            intent=intent,
            reason="sensitive_data_detected",
        )
        return {
            "status": "privacy_denied",
            "response": (
                "Không thể tạo truy vấn guideline đã khử định danh an toàn. "
                "Không có dữ liệu bệnh nhân nào được gửi sang nguồn guideline."
            ),
            "patient_evidence": patient_result.get("evidence", []),
            "guideline_evidence": [],
        }

    guideline_result = guideline_retriever(
        question=guideline_question,
        **guideline_options,
    )
    response = (
        "Dữ liệu bệnh án đã phân quyền:\n"
        f"{format_grounded_response(patient_result)}\n\n"
        "Nội dung guideline đã duyệt và còn hiệu lực:\n"
        f"{str(guideline_result.get('response') or '')}\n\n"
        f"{COMPOSITE_SAFETY_NOTE}"
    )
    status = (
        "success"
        if guideline_result.get("status") == "success"
        else "guideline_evidence_unavailable"
    )
    audit_event(
        "patient_guideline_workflow_completed",
        intent=intent,
        status=status,
        patient_fact_count=len(patient_facts),
        explicit_term_count=len(terms),
        guideline_status=guideline_result.get("status"),
    )
    return {
        "status": status,
        "response": response,
        "patient_evidence": patient_result.get("evidence", []),
        "guideline_evidence": guideline_result.get("evidence", []),
        "guideline_query": guideline_question,
        "intent": intent,
    }


def _validate_explicit_terms(
    question: str,
    explicit_terms: Sequence[str],
    patient_reference: str,
) -> Optional[List[str]]:
    """Accept only bounded terms copied exactly from the trusted question."""
    if (
        isinstance(explicit_terms, (str, bytes))
        or not isinstance(explicit_terms, Sequence)
        or len(explicit_terms) > MAX_EXPLICIT_TERMS
    ):
        return None

    question_folded = (question or "").casefold()
    accepted: List[str] = []
    seen = set()
    for raw_value in explicit_terms:
        if not isinstance(raw_value, str):
            return None
        value = raw_value.strip()
        folded = value.casefold()
        if (
            not value
            or len(value) > MAX_FACT_LENGTH
            or not SAFE_FACT_PATTERN.fullmatch(value)
            or not _contains_explicit_phrase(question_folded, folded)
            or _contains_patient_reference(value, patient_reference)
        ):
            return None
        if folded not in seen:
            seen.add(folded)
            accepted.append(value)
    return accepted


def _patient_description(field_name: str, value: str) -> str:
    if field_name == "patient_id":
        return f"bệnh nhân có patient_id: {value}"
    escaped = value.replace('"', "")
    return f'bệnh nhân "{escaped}"'


def _extract_allowed_facts(
    rows: Iterable[Dict[str, Any]], allowed_aliases: Sequence[str]
) -> List[str]:
    """Copy only scalar values under aliases allowlisted by the selected policy."""
    facts: List[str] = []
    seen = set()
    allowed = {alias.casefold() for alias in allowed_aliases}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        for field, raw_value in row.items():
            if (
                not isinstance(field, str)
                or field.casefold() not in allowed
                or not isinstance(raw_value, (str, int, float))
                or isinstance(raw_value, bool)
            ):
                continue
            value = str(raw_value).strip()
            folded = value.casefold()
            if (
                value
                and len(value) <= MAX_FACT_LENGTH
                and SAFE_FACT_PATTERN.fullmatch(value)
                and folded not in seen
            ):
                seen.add(folded)
                facts.append(value)
                if len(facts) >= MAX_FACTS:
                    return facts
    return facts


def _build_guideline_question(
    intent: str,
    policy: CompositePolicy,
    explicit_terms: Sequence[str],
    patient_facts: Sequence[str],
) -> str:
    if intent == "drug_interaction":
        values = _deduplicate([*explicit_terms, *patient_facts])
        return f"{policy.handoff_label}: {', '.join(values)}"
    return f"{policy.handoff_label}: {', '.join(patient_facts)}"


def _contains_explicit_phrase(text: str, phrase: str) -> bool:
    """Match a copied value without accepting a substring of another token."""
    return bool(re.search(
        rf"(?<!\w){re.escape(phrase)}(?!\w)", text, re.IGNORECASE
    ))


def _contains_patient_reference(value: str, patient_reference: str) -> bool:
    value_folded = value.casefold()
    reference_folded = patient_reference.casefold()
    return value_folded == reference_folded or (
        len(reference_folded) >= 3 and reference_folded in value_folded
    )


def _deduplicate(values: Sequence[str]) -> List[str]:
    result = []
    seen = set()
    for value in values:
        folded = value.casefold()
        if folded not in seen:
            seen.add(folded)
            result.append(value)
    return result


def _clarification(reason: str) -> Dict[str, Any]:
    audit_event("patient_guideline_clarification_required", reason=reason)
    return {
        "status": "needs_clarification",
        "response": COMPOSITE_CLARIFICATION,
        "patient_evidence": [],
        "guideline_evidence": [],
    }
