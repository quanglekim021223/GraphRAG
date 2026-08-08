"""Bounded patient-record plus curated-guideline workflow.

The outer agent may select this workflow for an explicit drug-interaction
question, but it cannot freely chain data-bearing tools. Patient facts stay in
the authorized GraphRAG path; only medication names are copied into the
de-identified curated-guideline query. Final output keeps patient and guideline
evidence separate and never asks an LLM to synthesize a clinical conclusion.
"""
import re
from typing import Any, Callable, Dict, Iterable, List, Sequence

from src.handlers.grounding_verifier import format_grounded_response
from src.handlers.medical_guideline_search import contains_sensitive_patient_data
from src.handlers.security_guardrails import (
    audit_event,
    extract_patient_reference,
)


MAX_TARGET_MEDICATIONS = 5
MAX_MEDICATION_NAME_LENGTH = 100
MEDICATION_FIELD_PATTERN = re.compile(
    r"(?:^|_)(?:medication|medicine|drug)(?:_|$)", re.IGNORECASE
)
SAFE_MEDICATION_PATTERN = re.compile(
    r"^[A-Za-zÀ-ỹ0-9][A-Za-zÀ-ỹ0-9 .,'()/+\-]*$"
)
COMPOSITE_CLARIFICATION = (
    "Vui lòng nêu rõ tên thuốc và bệnh nhân cần kiểm tra. Hệ thống chỉ đối "
    "chiếu thuốc được nêu rõ với thuốc đang được ghi nhận trong hồ sơ đã phân quyền."
)
NO_PATIENT_MEDICATIONS = (
    "Không tìm thấy tên thuốc có thể kiểm chứng trong kết quả bệnh án. "
    "Vui lòng kiểm tra lại bệnh nhân hoặc hỏi cụ thể hơn."
)
COMPOSITE_SAFETY_NOTE = (
    "Các dữ kiện bệnh án và nội dung guideline được hiển thị riêng; hệ thống "
    "không tự kết luận quan hệ nhân quả, chẩn đoán hoặc thay đổi điều trị."
)


def run_patient_guideline_workflow(
    question: str,
    target_medications: Sequence[str],
    doctor_id: str,
    graphrag: Any,
    guideline_retriever: Callable[..., Dict[str, Any]],
    guideline_options: Dict[str, Any],
) -> Dict[str, Any]:
    """Run one authorized GraphRAG lookup then one de-identified retrieval."""
    patient_reference = extract_patient_reference(question)
    if patient_reference is None:
        return _clarification("missing_patient_reference")

    reference_field, reference_value = patient_reference
    targets = _validate_target_medications(
        question, target_medications, reference_value
    )
    if not targets:
        return _clarification("missing_or_unverified_target_medication")

    patient_question = _patient_medication_question(
        reference_field, reference_value
    )
    patient_result = graphrag.run(patient_question, doctor_id)
    if patient_result.get("status") != "success":
        audit_event(
            "patient_guideline_patient_lookup_stopped",
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

    current_medications = _extract_medications(patient_result.get("result", []))
    if not current_medications:
        audit_event("patient_guideline_no_patient_medications", level="warning")
        return {
            "status": "needs_clarification",
            "response": NO_PATIENT_MEDICATIONS,
            "patient_evidence": patient_result.get("evidence", []),
            "guideline_evidence": [],
        }

    guideline_question = _build_guideline_question(targets, current_medications)
    if (
        any(
            _contains_patient_reference(name, reference_value)
            for name in [*targets, *current_medications]
        )
        or _contains_explicit_phrase(
            guideline_question.casefold(), reference_value.casefold()
        )
        or contains_sensitive_patient_data(guideline_question)
    ):
        audit_event(
            "patient_guideline_handoff_rejected",
            level="warning",
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
    patient_output = format_grounded_response(patient_result)
    guideline_output = str(guideline_result.get("response") or "")
    response = (
        "Dữ liệu bệnh án đã phân quyền:\n"
        f"{patient_output}\n\n"
        "Nội dung guideline đã duyệt và còn hiệu lực:\n"
        f"{guideline_output}\n\n"
        f"{COMPOSITE_SAFETY_NOTE}"
    )
    status = (
        "success"
        if guideline_result.get("status") == "success"
        else "guideline_evidence_unavailable"
    )
    audit_event(
        "patient_guideline_workflow_completed",
        status=status,
        patient_medication_count=len(current_medications),
        target_medication_count=len(targets),
        guideline_status=guideline_result.get("status"),
    )
    return {
        "status": status,
        "response": response,
        "patient_evidence": patient_result.get("evidence", []),
        "guideline_evidence": guideline_result.get("evidence", []),
        "guideline_query": guideline_question,
    }


def _validate_target_medications(
    question: str,
    target_medications: Sequence[str],
    patient_reference: str,
) -> List[str]:
    """Accept only bounded medication names copied from the current question."""
    if (
        isinstance(target_medications, (str, bytes))
        or not isinstance(target_medications, Sequence)
        or len(target_medications) > MAX_TARGET_MEDICATIONS
    ):
        return []

    question_folded = (question or "").casefold()
    accepted: List[str] = []
    seen = set()
    for raw_name in target_medications:
        if not isinstance(raw_name, str):
            return []
        name = raw_name.strip()
        folded = name.casefold()
        if (
            not name
            or len(name) > MAX_MEDICATION_NAME_LENGTH
            or not SAFE_MEDICATION_PATTERN.fullmatch(name)
            or not _contains_explicit_phrase(question_folded, folded)
            or _contains_patient_reference(name, patient_reference)
        ):
            return []
        if folded not in seen:
            seen.add(folded)
            accepted.append(name)
    return accepted


def _patient_medication_question(field_name: str, value: str) -> str:
    if field_name == "patient_id":
        return (
            "Liệt kê tên thuốc đang được ghi nhận cho bệnh nhân có "
            f"patient_id: {value}."
        )
    escaped = value.replace('"', "")
    return (
        "Liệt kê tên thuốc đang được ghi nhận cho bệnh nhân "
        f'"{escaped}".'
    )


def _extract_medications(rows: Iterable[Dict[str, Any]]) -> List[str]:
    medications: List[str] = []
    seen = set()
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        for field, value in row.items():
            if (
                not isinstance(field, str)
                or not MEDICATION_FIELD_PATTERN.search(field)
                or not isinstance(value, str)
            ):
                continue
            medication = value.strip()
            folded = medication.casefold()
            if (
                medication
                and len(medication) <= MAX_MEDICATION_NAME_LENGTH
                and SAFE_MEDICATION_PATTERN.fullmatch(medication)
                and folded not in seen
            ):
                seen.add(folded)
                medications.append(medication)
    return medications


def _build_guideline_question(
    targets: Sequence[str], current_medications: Sequence[str]
) -> str:
    medications = _deduplicate([*targets, *current_medications])
    return (
        "Drug interaction, contraindication, or medication safety guidance "
        "for concurrent use of: " + ", ".join(medications)
    )


def _contains_explicit_phrase(text: str, phrase: str) -> bool:
    """Match a copied name without accepting a substring of another token."""
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
