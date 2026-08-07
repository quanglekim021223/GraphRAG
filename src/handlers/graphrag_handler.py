"""
GraphRAG Handler module for Healthcare system.

This module implements the GraphRAG (Graph Retrieval-Augmented Generation) pipeline,
combining Neo4j graph database queries with LLM processing to provide accurate
healthcare information responses based on structured knowledge graphs.
"""
from typing import Dict, Any
from langsmith import Client
from src.helpers.logging_config import logger
from src.config.settings import Config
from src.handlers.grounding_verifier import (
    build_grounded_output,
    validate_template,
)
from src.handlers.graph_manager import GraphManager
from src.handlers.llm_manager import LLMManager
from src.handlers.security_guardrails import (
    AuthorizationScopeError,
    check_input_scope,
    check_prompt_injection,
    enforce_scope,
    validate_result,
)


class HealthcareGraphRAG:
    """Main GraphRAG system for healthcare data retrieval and question answering."""
    _instance = None
    MAX_CYPHER_RETRIES = 2

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(HealthcareGraphRAG, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the GraphRAG system with necessary components."""
        # Ngăn chặn khởi tạo lại nếu đã tồn tại
        if not hasattr(self, '_initialized'):
            self.config = Config()  # Sử dụng instance singleton của Config
            self.graph_manager = GraphManager(self.config)
            self.llm_manager = LLMManager(self.config)
            self.schema = self.graph_manager.get_schema()

            try:
                self.langsmith_client = Client()
                logger.info("LangSmith client initialized successfully.")
            except Exception as e:  # pylint: disable=broad-exception-caught
                # Catching all exceptions is acceptable here since this is optional functionality
                logger.warning(
                    "LangSmith client initialization failed: %s", str(e))

            self._initialized = True

    @staticmethod
    def _clarification_result(query: str, reason: str, attempts: int) -> Dict[str, Any]:
        """Return a safe response that lets the outer ReAct agent ask the user."""
        if reason == "empty_result":
            response = (
                "Tôi chưa tìm thấy dữ liệu phù hợp. Vui lòng cung cấp thêm "
                "thông tin định danh như họ tên đầy đủ, mã bệnh nhân, bệnh viện "
                "hoặc diễn đạt cụ thể hơn mối quan hệ cần tìm."
            )
        else:
            response = (
                "Tôi chưa thể tạo truy vấn dữ liệu phù hợp sau một số lần thử. "
                "Vui lòng diễn đạt câu hỏi cụ thể hơn hoặc bổ sung tên, mã định "
                "danh và mối quan hệ cần tra cứu."
            )

        return {
            "status": "needs_clarification",
            "query": query,
            "response": response,
            "reason": reason,
            "attempts": attempts,
        }

    @staticmethod
    def _guardrail_result(status: str, reason: str, response: str) -> Dict[str, Any]:
        """Return a controlled response without exposing protected data."""
        return {
            "status": status,
            "query": None,
            "response": response,
            "reason": reason,
            "attempts": 0,
            "evidence": [],
        }

    def run(self, question: str, doctor_id: str) -> Dict[str, Any]:
        """
        Run the GraphRAG pipeline on a question.

        Args:
            question: User query string
            doctor_id: Authenticated doctor identity from application context

        Returns:
            Dict containing the query and response
        """
        query = None
        if not doctor_id:
            return self._guardrail_result(
                "authorization_denied",
                "missing_doctor_identity",
                "Không thể xác định danh tính bác sĩ đã đăng nhập.",
            )
        if check_prompt_injection(question):
            return self._guardrail_result(
                "manual_review",
                "suspected_prompt_injection",
                "Yêu cầu này cần được kiểm tra thủ công trước khi truy cập dữ liệu.",
            )
        if not check_input_scope(
            question,
            doctor_id,
            self.graph_manager.patient_reference_in_scope,
        ):
            return self._guardrail_result(
                "authorization_denied",
                "patient_out_of_scope_or_unverified",
                "Không thể xác nhận bệnh nhân thuộc phạm vi phụ trách của bác sĩ.",
            )

        try:
            generation = self.llm_manager.generate_cypher_query(
                question, self.schema
            )
            generated_query = generation.cypher
            response_template = generation.response_template
            if not validate_template(response_template, generated_query):
                logger.warning(
                    "Generated response template was rejected; using "
                    "deterministic rendering"
                )
                response_template = None
            try:
                query = enforce_scope(generated_query, doctor_id)
            except AuthorizationScopeError:
                return self._guardrail_result(
                    "authorization_denied",
                    "unsafe_cypher_scope",
                    "Truy vấn bị từ chối vì không thể áp dụng phạm vi bác sĩ an toàn.",
                )

            for retry_count in range(self.MAX_CYPHER_RETRIES + 1):
                try:
                    parameters = {"doctor_id": doctor_id}
                    self.graph_manager.explain_query(query, parameters)
                    query_result = self.graph_manager.execute_query(
                        query, parameters
                    )
                except ValueError as error:
                    logger.warning(
                        "Cypher attempt %s/%s failed: %s",
                        retry_count + 1,
                        self.MAX_CYPHER_RETRIES + 1,
                        str(error),
                    )
                    if retry_count >= self.MAX_CYPHER_RETRIES:
                        return self._clarification_result(
                            query, "max_retries_exceeded", retry_count + 1
                        )

                    repaired = self.llm_manager.repair_cypher_query(
                        question=question,
                        schema=self.schema,
                        invalid_query=generated_query,
                        diagnostic=str(error),
                    )
                    repaired_query = repaired.cypher
                    if response_template and validate_template(
                        response_template, repaired_query
                    ):
                        pass
                    elif validate_template(
                        repaired.response_template, repaired_query
                    ):
                        response_template = repaired.response_template
                    else:
                        logger.warning(
                            "Old and repaired response templates were rejected; using "
                            "deterministic rendering"
                        )
                        response_template = None
                    generated_query = repaired_query
                    try:
                        query = enforce_scope(generated_query, doctor_id)
                    except AuthorizationScopeError:
                        return self._guardrail_result(
                            "authorization_denied",
                            "unsafe_repaired_cypher_scope",
                            "Truy vấn sửa lại bị từ chối vì phạm vi bác sĩ không an toàn.",
                        )
                    continue

                if not query_result:
                    logger.info("Cypher query returned no records: %s", query)
                    return self._clarification_result(
                        query, "empty_result", retry_count + 1
                    )

                validation = validate_result(
                    query_result,
                    query,
                    question,
                    max_rows=self.config.max_result_rows,
                )
                if not validation.valid:
                    return {
                        "status": "validation_failed",
                        "query": query,
                        "response": validation.user_message,
                        "reason": validation.reason,
                        "attempts": retry_count + 1,
                        "evidence": [],
                    }

                grounded_output = build_grounded_output(
                    validation.rows,
                    response_template,
                    validation.field_warnings,
                )
                if not grounded_output["grounded"]:
                    logger.warning(
                        "Response abstained because grounding verification failed: %s",
                        grounded_output["reason"],
                    )
                    return {
                        "status": "abstained",
                        "query": query,
                        "result": query_result,
                        "response": grounded_output["answer"],
                        "evidence": [],
                        "reason": grounded_output["reason"],
                        "attempts": retry_count + 1,
                    }

                result = {
                    "status": "success",
                    "query": query,
                    "result": query_result,
                    "response": grounded_output["answer"],
                    "evidence": grounded_output["evidence"],
                    "omitted_fields": grounded_output["omitted_fields"],
                    "flagged_for_review": validation.flagged_for_review,
                    "possible_semantic_mismatch": (
                        validation.possible_semantic_mismatch
                    ),
                    "warnings": validation.warnings,
                    "attempts": retry_count + 1,
                }
                if validation.possible_semantic_mismatch:
                    result["response"] += (
                        "\n\nCảnh báo: truy vấn có thể chưa phản ánh đầy đủ "
                        "ý 'mới nhất/gần nhất'; vui lòng xác minh lại."
                    )
                logger.info(
                    "Successfully processed query '%s' after %s attempt(s)",
                    question,
                    retry_count + 1,
                )
                return result

            return self._clarification_result(
                query, "max_retries_exceeded", self.MAX_CYPHER_RETRIES + 1
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Unexpected error in pipeline for '%s': %s",
                         question, str(e), exc_info=True)
            return self._clarification_result(
                query, "pipeline_error", 1
            )
