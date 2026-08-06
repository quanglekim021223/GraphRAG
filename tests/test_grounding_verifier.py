"""Tests for deterministic, fail-closed Neo4j result rendering."""
import unittest

from src.handlers.grounding_verifier import (
    build_grounded_output,
    format_grounded_response,
    render_answer,
    select_controlled_agent_response,
    validate_template,
)


class GroundingVerifierTests(unittest.TestCase):
    """Only scalar values returned by Neo4j may reach the answer."""

    def test_scalar_values_are_rendered_without_type_conversion(self):
        result = build_grounded_output(
            [{"patient_name": "Alice", "patient_age": 30}]
        )

        self.assertTrue(result["grounded"])
        self.assertIn("Patient name: Alice", result["answer"])
        self.assertIn("Patient age: 30", result["answer"])
        self.assertEqual(30, result["evidence"][0]["sources"][1]["value"])

    def test_valid_template_renders_natural_grounded_answer(self):
        cypher = (
            "MATCH (p:Patient) RETURN p.name AS patient_name, "
            "p.age AS patient_age LIMIT 5"
        )
        template = "Bệnh nhân {patient_name} năm nay {patient_age} tuổi."

        self.assertTrue(validate_template(template, cypher))
        answer, evidence = render_answer(
            [{"patient_name": "Alice", "patient_age": 30}], template
        )

        self.assertEqual("- Bệnh nhân Alice năm nay 30 tuổi. [E1]", answer)
        self.assertEqual(
            ["patient_name", "patient_age"],
            [source["field"] for source in evidence[0]["sources"]],
        )

    def test_unknown_placeholder_is_rejected_and_falls_back(self):
        cypher = "MATCH (p:Patient) RETURN p.name AS patient_name LIMIT 5"
        template = "Bệnh nhân {disease_name}."

        self.assertFalse(validate_template(template, cypher))
        result = build_grounded_output(
            [{"patient_name": "Alice"}],
            template if validate_template(template, cypher) else None,
        )

        self.assertIn("Patient name: Alice", result["answer"])

    def test_hardcoded_number_is_rejected_and_falls_back(self):
        cypher = "MATCH (p:Patient) RETURN p.age AS patient_age LIMIT 5"
        template = "Bệnh nhân năm nay 30 tuổi, dữ liệu là {patient_age}."

        self.assertFalse(validate_template(template, cypher))
        result = build_grounded_output(
            [{"patient_age": 30}],
            template if validate_template(template, cypher) else None,
        )

        self.assertIn("Patient age: 30", result["answer"])

    def test_multi_row_template_is_applied_once_per_row(self):
        template = "Bệnh nhân {patient_name}."

        answer, evidence = render_answer(
            [{"patient_name": "Alice"}, {"patient_name": "Bob"}],
            template,
        )

        self.assertIn("- Bệnh nhân Alice. [E1]", answer)
        self.assertIn("- Bệnh nhân Bob. [E2]", answer)
        self.assertEqual(2, len(evidence))

    def test_missing_template_field_falls_back_for_only_that_row(self):
        template = "Bệnh nhân {patient_name} năm nay {patient_age} tuổi."

        answer, evidence = render_answer(
            [
                {"patient_name": "Alice", "patient_age": 30},
                {"patient_name": "Bob"},
            ],
            template,
        )

        self.assertIn("Bệnh nhân Alice năm nay 30 tuổi", answer)
        self.assertIn("Patient name: Bob", answer)
        self.assertEqual(["patient_name"], [
            source["field"] for source in evidence[1]["sources"]
        ])

    def test_cypher_literal_cannot_be_copied_into_template(self):
        cypher = (
            "MATCH (p:Patient) WHERE p.name = 'Alice' "
            "RETURN p.name AS patient_name LIMIT 5"
        )

        self.assertFalse(validate_template("Alice là {patient_name}.", cypher))

    def test_string_case_whitespace_and_date_are_not_normalized(self):
        result = build_grounded_output(
            [{"patient_name": "  ALICE  ", "test_date": "2026-08-06"}]
        )

        self.assertIn("Patient name:   ALICE  ", result["answer"])
        self.assertIn("Test date: 2026-08-06", result["answer"])

    def test_aggregate_is_rendered_only_from_cypher_scalar(self):
        result = build_grounded_output(
            [
                {
                    "patient_name": "Alice",
                    "abnormal_test_count": 3,
                    "period_year": 2026,
                }
            ]
        )

        self.assertIn("Abnormal test count: 3", result["answer"])
        self.assertIn("Period year: 2026", result["answer"])
        self.assertEqual(3, result["evidence"][0]["sources"][1]["value"])

    def test_unsupported_field_is_omitted_and_disclosed(self):
        result = build_grounded_output(
            [{"patient_name": "Alice", "raw_node": {"age": 30}}]
        )

        self.assertTrue(result["grounded"])
        self.assertEqual(1, result["omitted_fields"])
        self.assertIn("1 trường đã được ẩn", result["answer"])
        self.assertNotIn("raw node", result["answer"].lower())

    def test_all_unsupported_fields_cause_abstention(self):
        result = build_grounded_output([{"raw_node": {"age": 30}}])

        self.assertFalse(result["grounded"])
        self.assertEqual("unsupported_result_shape", result["reason"])
        self.assertEqual([], result["evidence"])

    def test_multiple_rows_get_separate_evidence_ids(self):
        result = build_grounded_output(
            [{"name": "Alice"}, {"name": "Bob"}]
        )

        self.assertIn("[E1]", result["answer"])
        self.assertIn("[E2]", result["answer"])
        self.assertEqual(2, len(result["evidence"]))

    def test_verified_evidence_is_formatted_for_react(self):
        verified = build_grounded_output([{"patient_age": 30}])
        tool_result = {
            "response": verified["answer"],
            "evidence": verified["evidence"],
        }

        formatted = format_grounded_response(tool_result)

        self.assertIn("[E1]", formatted)
        self.assertIn("row 0.patient_age=30", formatted)

    def test_rag_tool_output_cannot_be_rewritten_by_react(self):
        full_response = {
            "messages": [
                {"type": "human", "content": "Alice bao nhiêu tuổi?"},
                {
                    "type": "tool",
                    "name": "rag_tool",
                    "content": "- Patient age: 30. [E1]",
                },
                {
                    "type": "ai",
                    "content": "Alice is 30 years old and very healthy. [E1]",
                },
            ]
        }

        selected = select_controlled_agent_response(full_response)

        self.assertEqual("- Patient age: 30. [E1]", selected)

    def test_previous_turn_rag_output_is_not_reused(self):
        full_response = {
            "messages": [
                {
                    "type": "tool",
                    "name": "rag_tool",
                    "content": "old grounded result",
                },
                {"type": "human", "content": "Huyết áp là gì?"},
                {
                    "type": "tool",
                    "name": "llm_tool",
                    "content": "general tool result",
                },
                {"type": "ai", "content": "general final answer"},
            ]
        }

        selected = select_controlled_agent_response(full_response)

        self.assertEqual("general final answer", selected)


if __name__ == "__main__":
    unittest.main()
