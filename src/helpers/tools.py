"""
Tools module for Healthcare GraphRAG system.

This module provides specialized tools used by the agent system, including RAG lookup
with Neo4j database and general LLM response generation. It handles routing between
knowledge database lookups and general medical knowledge responses.
"""
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from src.helpers.logging_config import logger
from src.handlers.grounding_verifier import format_grounded_response
from src.handlers.graphrag_handler import HealthcareGraphRAG
from src.helpers.llm_initializer import get_llm

# Initialize singleton instances
graphrag_instance = HealthcareGraphRAG()
llm = get_llm()

# Last executed Cypher query for debugging/display
LAST_QUERY = None


def get_last_query():
    """
    Get the last executed Cypher query.

    Returns:
        str: The most recent Cypher query or None if no query has been executed
    """
    return LAST_QUERY


def set_last_query(query):
    """
    Set the last executed Cypher query.

    Args:
        query: The Cypher query string to store
    """
    global LAST_QUERY
    LAST_QUERY = query


@tool
def rag_tool(question: str) -> str:
    """Use this tool to query specific healthcare data from the database."""
    try:
        result = graphrag_instance.run(question)
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
        return f"Error: {str(e)}"


@tool
def llm_tool(question: str) -> str:
    """Use this tool to provide general medical knowledge or when specific data is not available."""
    try:
        # Prepare a general knowledge prompt when database lookup fails
        general_prompt = (
            f"You are a healthcare assistant. User asked: '{question}'. "
            "No specific data was found in the database. "
            "Provide a general answer based on common medical knowledge and "
            "append a follow-up question like 'Do you have any more questions?' "
            "or 'Is this the information you were looking for?'"
        )
        return llm.invoke([
            SystemMessage(content="You are a healthcare assistant."),
            HumanMessage(content=general_prompt)
        ]).content.strip()
    except Exception as e:  # pylint: disable=broad-exception-caught
        # Broad exception is necessary here as this is a fallback tool
        logger.error("LLM tool error: %s", str(e))
        return f"Error generating response: {str(e)}"
