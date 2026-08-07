"""
Tools module for Healthcare GraphRAG system.

This module provides grounded Neo4j lookup and allowlisted medical-guideline
retrieval. Neither tool lets the outer ReAct agent rewrite controlled output.
"""
from contextvars import ContextVar
from typing import Optional

from langchain_core.tools import tool
from src.config.settings import Config
from src.handlers.medical_guideline_search import search_medical_guidelines
from src.helpers.logging_config import logger
from src.handlers.grounding_verifier import (
    format_grounded_response,
)
from src.handlers.graphrag_handler import HealthcareGraphRAG
from src.helpers.security_context import claim_request_tool, get_current_doctor_id


TOOL_CHAIN_BLOCKED_RESPONSE = (
    "Hệ thống đã chặn việc gọi nhiều nguồn dữ liệu trong cùng một lượt để "
    "tránh trộn dữ liệu bệnh án với nội dung tìm kiếm bên ngoài."
)

# Initialize singleton instances
graphrag_instance = HealthcareGraphRAG()

# Request-local Cypher metadata; a process global can leak across doctors.
_LAST_QUERY: ContextVar[Optional[str]] = ContextVar("last_query", default=None)


def get_last_query():
    """
    Get the last executed Cypher query.

    Returns:
        str: The most recent Cypher query or None if no query has been executed
    """
    return _LAST_QUERY.get()


def set_last_query(query):
    """
    Set the last executed Cypher query.

    Args:
        query: The Cypher query string to store
    """
    _LAST_QUERY.set(query)


@tool
def rag_tool(question: str) -> str:
    """Use this tool to query specific healthcare data from the database."""
    try:
        if not claim_request_tool("rag_tool"):
            return TOOL_CHAIN_BLOCKED_RESPONSE
        result = graphrag_instance.run(question, get_current_doctor_id())
        logger.info("GraphRAG completed with status: %s", result.get("status"))

        # Lưu query nếu có
        if isinstance(result, dict) and "query" in result:
            set_last_query(result["query"])

        # Return only controlled output; never expose raw Cypher as a fallback.
        if isinstance(result, dict) and "response" in result:
            return format_grounded_response(result)

        return "No information found"
    except Exception as e:  # pylint: disable=broad-exception-caught
        # Broad exception is necessary here as this is a fallback tool
        logger.error("GraphRAG error: %s", str(e))
        return "Không thể truy cập dữ liệu khi chưa xác định danh tính bác sĩ."


@tool
def medical_guideline_tool(question: str) -> str:
    """Retrieve general medical guidance from approved public sources only."""
    if not claim_request_tool("medical_guideline_tool"):
        return TOOL_CHAIN_BLOCKED_RESPONSE
    config = Config()
    result = search_medical_guidelines(
        question=question,
        api_key=config.tavily_api_key,
        max_results=config.medical_search_max_results,
        min_score=config.medical_search_min_score,
    )
    return str(result["response"])
