"""
GraphRAG Handler module for Healthcare system.

This module implements the GraphRAG (Graph Retrieval-Augmented Generation) pipeline,
combining Neo4j graph database queries with LLM processing to provide accurate
healthcare information responses based on structured knowledge graphs.
"""
from typing import Dict, Any
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableSequence
from langsmith import Client
from src.helpers.logging_config import logger
from src.config.settings import Config
from src.handlers.graph_manager import GraphManager
from src.handlers.llm_manager import LLMManager


class HealthcareGraphRAG:
    """Main GraphRAG system for healthcare data retrieval and question answering."""
    _instance = None

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

            # Khởi tạo pipeline một lần trong __init__
            self.pipeline = self._create_pipeline()

            self._initialized = True

    def _create_pipeline(self) -> RunnableSequence:
        """
        Create the GraphRAG processing pipeline.

        Returns:
            RunnableSequence: The complete GraphRAG pipeline
        """
        pipeline = (
            {"question": RunnablePassthrough(), "schema": RunnableLambda(
                lambda _: self.schema)}
            | RunnableLambda(lambda x: {
                "question": x["question"],
                "schema": x["schema"],
                "query": self.llm_manager.generate_cypher_query(x["question"], x["schema"])
            }).with_config(run_name="GenerateCypherQuery")
            | RunnableLambda(lambda x: {
                "question": x["question"],
                "schema": x["schema"],
                "query": self.llm_manager.validate_cypher_query(x["query"], x["schema"])
            }).with_config(run_name="ValidateCypherQuery")
            | RunnableLambda(lambda x: {
                "question": x["question"],
                "schema": x["schema"],
                "query": x["query"],
                "result": self.graph_manager.execute_query(x["query"])
            }).with_config(run_name="ExecuteCypherQuery")
            | RunnableLambda(lambda x: {
                "query": x["query"],
                "result": x["result"],
                "response": self.llm_manager.generate_response(x["question"], x["result"])
            }).with_config(run_name="GenerateResponse")
        )
        return pipeline

    def run(self, question: str) -> Dict[str, Any]:
        """
        Run the GraphRAG pipeline on a question.

        Args:
            question: User query string

        Returns:
            Dict containing the query and response
        """
        try:
            # Sử dụng pipeline đã khởi tạo thay vì tạo mới
            result = self.pipeline.invoke(question)
            logger.info("Successfully processed query: '%s' with result: %s",
                        question, result)
            return result
        except ValueError as e:
            logger.error("Pipeline failed for '%s': %s", question, str(e))
            return {"query": None, "response": f"Error: {str(e)}"}
        except Exception as e:  # pylint: disable=broad-exception-caught
            # This is the top-level error handler for the pipeline, so broad exception is appropriate
            logger.error("Unexpected error in pipeline for '%s': %s",
                         question, str(e), exc_info=True)
            return {"query": None, "response": f"Error: {str(e)}"}
