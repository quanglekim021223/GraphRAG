from typing import Dict, Any, List
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableSequence
from langsmith import Client
from src.helpers.logging_config import logger
from src.config.settings import Config
from src.handlers.graph_manager import GraphManager
from src.handlers.llm_manager import LLMManager


class HealthcareGraphRAG:
    """Main GraphRAG system for healthcare data retrieval and question answering."""

    def __init__(self, config: Config) -> None:
        """Initialize the GraphRAG system with configuration."""
        self.config = config
        self.graph_manager = GraphManager(config)
        self.llm_manager = LLMManager(config)
        self.schema = self.graph_manager.get_schema()

        try:
            self.langsmith_client = Client()
            logger.info("LangSmith client initialized successfully.")
        except Exception as e:
            logger.warning(f"LangSmith client initialization failed: {str(e)}")

    def get_pipeline(self) -> RunnableSequence:
        """Create the GraphRAG processing pipeline."""
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
                "query": x["result"],
                "response": self.llm_manager.generate_response(x["question"], x["result"])
            }).with_config(run_name="GenerateResponse")
        )
        return pipeline

    def run(self, question: str) -> Dict[str, Any]:
        """Run the GraphRAG pipeline on a question."""
        try:
            pipeline = self.get_pipeline()
            result = pipeline.invoke(question)
            logger.info(
                f"Successfully processed query: '{question}' with result: {result}")
            return result
        except ValueError as e:
            logger.error(f"Pipeline failed for '{question}': {str(e)}")
            return {"query": None, "response": f"Error: {str(e)}"}
        except Exception as e:
            logger.error(
                f"Unexpected error in pipeline for '{question}': {str(e)}", exc_info=True)
            return {"query": None, "response": f"Error: {str(e)}"}
