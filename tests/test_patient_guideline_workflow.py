"""Tests for the bounded patient-record plus guideline workflow."""
import unittest

from src.handlers.patient_guideline_workflow import (
    run_patient_guideline_workflow,
)


class FakeGraphRAG:
    """Return controlled patient medication facts without external services."""

    def __init__(self, result=None):
        self.calls = []
        self.result = result or {
            "status": "success",
            "response": "- Alice đang dùng Warfarin. [E1]",
            "result": [
                {"patient_name": "Alice", "medication_name": "Warfarin"}
            ],
            "evidence": [
                {
                    "id": "E1",
                    "claim": "Alice đang dùng Warfarin.",
                    "sources": [
                        {
                            "row": 0,
                            "field": "medication_name",
                            "value": "Warfarin",
                        }
                    ],
                }
            ],
        }

    def run(self, question, doctor_id):
        self.calls.append((question, doctor_id))
        return self.result


class FakeGuidelineRetriever:
    """Record the de-identified query and return one reviewed section."""

    def __init__(self, status="success"):
        self.calls = []
        self.status = status

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.status != "success":
            return {
                "status": self.status,
                "response": "Không tìm thấy section đủ liên quan.",
            }
        return {
            "status": "success",
            "response": "[G1] Guideline tương tác thuốc đã duyệt.",
            "evidence": [{"id": "G1", "section_id": "section-1"}],
        }


class PatientGuidelineWorkflowTests(unittest.TestCase):
    """Keep patient identity local and evidence namespaces separate."""

    def test_composite_workflow_uses_deidentified_guideline_query(self):
        graph = FakeGraphRAG()
        retriever = FakeGuidelineRetriever()

        result = run_patient_guideline_workflow(
            question=(
                "Ibuprofen có tương tác với thuốc bệnh nhân \"Alice\" "
                "đang dùng không?"
            ),
            intent="drug_interaction",
            explicit_terms=["Ibuprofen"],
            doctor_id="doctor-1",
            graphrag=graph,
            guideline_retriever=retriever,
            guideline_options={"postgres_uri": "postgresql://test/test"},
        )

        self.assertEqual("success", result["status"])
        self.assertEqual(1, len(graph.calls))
        self.assertEqual("doctor-1", graph.calls[0][1])
        self.assertIn('bệnh nhân "Alice"', graph.calls[0][0])
        guideline_question = retriever.calls[0]["question"]
        self.assertIn("Ibuprofen", guideline_question)
        self.assertIn("Warfarin", guideline_question)
        self.assertNotIn("Alice", guideline_question)
        self.assertIn("[E1]", result["response"])
        self.assertIn("[G1]", result["response"])
        self.assertEqual("E1", result["patient_evidence"][0]["id"])
        self.assertEqual("G1", result["guideline_evidence"][0]["id"])

    def test_target_must_be_copied_from_current_question(self):
        graph = FakeGraphRAG()
        retriever = FakeGuidelineRetriever()

        result = run_patient_guideline_workflow(
            question='Thuốc này có ảnh hưởng đến bệnh nhân "Alice" không?',
            intent="drug_interaction",
            explicit_terms=["Ibuprofen"],
            doctor_id="doctor-1",
            graphrag=graph,
            guideline_retriever=retriever,
            guideline_options={},
        )

        self.assertEqual("needs_clarification", result["status"])
        self.assertEqual([], graph.calls)
        self.assertEqual([], retriever.calls)

    def test_patient_name_cannot_be_smuggled_as_medication(self):
        graph = FakeGraphRAG()
        retriever = FakeGuidelineRetriever()

        result = run_patient_guideline_workflow(
            question='Kiểm tra Alice cho bệnh nhân "Alice".',
            intent="drug_interaction",
            explicit_terms=["Alice"],
            doctor_id="doctor-1",
            graphrag=graph,
            guideline_retriever=retriever,
            guideline_options={},
        )

        self.assertEqual("needs_clarification", result["status"])
        self.assertEqual([], graph.calls)
        self.assertEqual([], retriever.calls)

    def test_single_letter_patient_reference_does_not_match_drug_substring(self):
        graph = FakeGraphRAG(
            {
                "status": "success",
                "response": "- Bệnh nhân A đang dùng Warfarin. [E1]",
                "result": [
                    {"patient_name": "A", "medication_name": "Warfarin"}
                ],
                "evidence": [],
            }
        )
        retriever = FakeGuidelineRetriever()

        result = run_patient_guideline_workflow(
            question="Ibuprofen có ảnh hưởng đến bệnh nhân A không?",
            intent="drug_interaction",
            explicit_terms=["Ibuprofen"],
            doctor_id="doctor-1",
            graphrag=graph,
            guideline_retriever=retriever,
            guideline_options={},
        )

        self.assertEqual("success", result["status"])
        self.assertNotIn("patient A", retriever.calls[0]["question"])

    def test_missing_patient_reference_stops_before_retrieval(self):
        graph = FakeGraphRAG()
        retriever = FakeGuidelineRetriever()

        result = run_patient_guideline_workflow(
            question="Ibuprofen có tương tác với Warfarin không?",
            intent="drug_interaction",
            explicit_terms=["Ibuprofen"],
            doctor_id="doctor-1",
            graphrag=graph,
            guideline_retriever=retriever,
            guideline_options={},
        )

        self.assertEqual("needs_clarification", result["status"])
        self.assertEqual([], graph.calls)
        self.assertEqual([], retriever.calls)

    def test_missing_patient_medication_stops_before_guideline(self):
        graph = FakeGraphRAG(
            {
                "status": "success",
                "response": "- Bệnh nhân Alice. [E1]",
                "result": [{"patient_name": "Alice"}],
                "evidence": [],
            }
        )
        retriever = FakeGuidelineRetriever()

        result = run_patient_guideline_workflow(
            question='Ibuprofen có ảnh hưởng bệnh nhân "Alice" không?',
            intent="drug_interaction",
            explicit_terms=["Ibuprofen"],
            doctor_id="doctor-1",
            graphrag=graph,
            guideline_retriever=retriever,
            guideline_options={},
        )

        self.assertEqual("needs_clarification", result["status"])
        self.assertEqual([], retriever.calls)

    def test_guideline_failure_is_disclosed_without_synthesis(self):
        graph = FakeGraphRAG()
        retriever = FakeGuidelineRetriever(status="not_found")

        result = run_patient_guideline_workflow(
            question='Ibuprofen có ảnh hưởng bệnh nhân "Alice" không?',
            intent="drug_interaction",
            explicit_terms=["Ibuprofen"],
            doctor_id="doctor-1",
            graphrag=graph,
            guideline_retriever=retriever,
            guideline_options={},
        )

        self.assertEqual("guideline_evidence_unavailable", result["status"])
        self.assertIn("Không tìm thấy section", result["response"])
        self.assertIn("không tự kết luận", result["response"])

    def test_disease_guideline_uses_only_condition_alias(self):
        graph = FakeGraphRAG(
            {
                "status": "success",
                "response": "- Alice có Hypertension. [E1]",
                "result": [{
                    "patient_name": "Alice",
                    "disease_name": "Hypertension",
                    "hospital_name": "City Hospital",
                }],
                "evidence": [{"id": "E1", "sources": []}],
            }
        )
        retriever = FakeGuidelineRetriever()

        result = run_patient_guideline_workflow(
            question='Guideline nào liên quan đến bệnh nền của bệnh nhân "Alice"?',
            intent="disease_guideline",
            explicit_terms=[],
            doctor_id="doctor-1",
            graphrag=graph,
            guideline_retriever=retriever,
            guideline_options={},
        )

        self.assertEqual("success", result["status"])
        guideline_question = retriever.calls[0]["question"]
        self.assertIn("Hypertension", guideline_question)
        self.assertNotIn("Alice", guideline_question)
        self.assertNotIn("City Hospital", guideline_question)

    def test_blood_type_compatibility_has_bounded_handoff(self):
        graph = FakeGraphRAG(
            {
                "status": "success",
                "response": "- Nhóm máu được ghi nhận là O+. [E1]",
                "result": [{"blood_type": "O+", "room_number": 101}],
                "evidence": [{"id": "E1", "sources": []}],
            }
        )
        retriever = FakeGuidelineRetriever()

        result = run_patient_guideline_workflow(
            question='Nhóm máu của bệnh nhân "Alice" tương thích truyền máu thế nào?',
            intent="blood_type_compatibility",
            explicit_terms=[],
            doctor_id="doctor-1",
            graphrag=graph,
            guideline_retriever=retriever,
            guideline_options={},
        )

        self.assertEqual("success", result["status"])
        guideline_question = retriever.calls[0]["question"]
        self.assertIn("O+", guideline_question)
        self.assertNotIn("101", guideline_question)

    def test_test_result_intent_is_rejected_until_schema_can_ground_it(self):
        graph = FakeGraphRAG()
        retriever = FakeGuidelineRetriever()

        result = run_patient_guideline_workflow(
            question='Kết quả xét nghiệm INR của bệnh nhân "Alice" có ý nghĩa gì?',
            intent="test_result_guideline",
            explicit_terms=["INR"],
            doctor_id="doctor-1",
            graphrag=graph,
            guideline_retriever=retriever,
            guideline_options={},
        )

        self.assertEqual("needs_clarification", result["status"])
        self.assertEqual([], graph.calls)
        self.assertEqual([], retriever.calls)

    def test_policy_rejects_admin_only_rows(self):
        graph = FakeGraphRAG(
            {
                "status": "success",
                "response": "- Hospital: City Hospital. [E1]",
                "result": [{
                    "hospital_name": "City Hospital",
                    "billing_amount": 1200,
                }],
                "evidence": [],
            }
        )
        retriever = FakeGuidelineRetriever()

        result = run_patient_guideline_workflow(
            question='Guideline nào liên quan đến bệnh nền của bệnh nhân "Alice"?',
            intent="disease_guideline",
            explicit_terms=[],
            doctor_id="doctor-1",
            graphrag=graph,
            guideline_retriever=retriever,
            guideline_options={},
        )

        self.assertEqual("needs_clarification", result["status"])
        self.assertEqual([], retriever.calls)

    def test_unknown_intent_fails_before_patient_lookup(self):
        graph = FakeGraphRAG()
        retriever = FakeGuidelineRetriever()

        result = run_patient_guideline_workflow(
            question='Thông tin của bệnh nhân "Alice"?',
            intent="hospital_guideline",
            explicit_terms=[],
            doctor_id="doctor-1",
            graphrag=graph,
            guideline_retriever=retriever,
            guideline_options={},
        )

        self.assertEqual("needs_clarification", result["status"])
        self.assertEqual([], graph.calls)
        self.assertEqual([], retriever.calls)


if __name__ == "__main__":
    unittest.main()
