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

    def run(self, question: str) -> Dict[str, Any]:
        """
        Run the GraphRAG pipeline on a question.

        Args:
            question: User query string

        Returns:
            Dict containing the query and response
        """
        query = None
        try:
            generation = self.llm_manager.generate_cypher_query(
                question, self.schema
            )
            query = generation.cypher
            response_template = generation.response_template
            if not validate_template(response_template, query):
                logger.warning(
                    "Generated response template was rejected; using "
                    "deterministic rendering"
                )
                response_template = None

            for retry_count in range(self.MAX_CYPHER_RETRIES + 1):
                try:
                    self.graph_manager.explain_query(query)
                    query_result = self.graph_manager.execute_query(query)
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

                    generation = self.llm_manager.repair_cypher_query(
                        question=question,
                        schema=self.schema,
                        invalid_query=query,
                        diagnostic=str(error),
                    )
                    query = generation.cypher
                    response_template = generation.response_template
                    if not validate_template(response_template, query):
                        logger.warning(
                            "Repaired response template was rejected; using "
                            "deterministic rendering"
                        )
                        response_template = None
                    continue

                if not query_result:
                    logger.info("Cypher query returned no records: %s", query)
                    return self._clarification_result(
                        query, "empty_result", retry_count + 1
                    )

                grounded_output = build_grounded_output(
                    query_result, response_template
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
                    "attempts": retry_count + 1,
                }
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
