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
            target_medications=["Ibuprofen"],
            doctor_id="doctor-1",
            graphrag=graph,
            guideline_retriever=retriever,
            guideline_options={"database_path": "test.sqlite3"},
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
            target_medications=["Ibuprofen"],
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
            target_medications=["Alice"],
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
            target_medications=["Ibuprofen"],
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
            target_medications=["Ibuprofen"],
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
            target_medications=["Ibuprofen"],
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
            target_medications=["Ibuprofen"],
            doctor_id="doctor-1",
            graphrag=graph,
            guideline_retriever=retriever,
            guideline_options={},
        )

        self.assertEqual("guideline_evidence_unavailable", result["status"])
        self.assertIn("Không tìm thấy section", result["response"])
        self.assertIn("không tự kết luận", result["response"])


if __name__ == "__main__":
    unittest.main()
