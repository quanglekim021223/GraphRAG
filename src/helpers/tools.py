"""
Tools module for Healthcare GraphRAG system.

This module provides specialized tools used by the agent system, including RAG lookup
with Neo4j database and general LLM response generation. It handles routing between
knowledge database lookups and general medical knowledge responses.
"""
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from src.helpers.logging_config import logger
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
        logger.info("Raw GraphRAG result: %s", result)

        # Lưu query nếu có
        if isinstance(result, dict) and "query" in result:
            set_last_query(result["query"])

        # Xử lý response
        if isinstance(result, dict) and "response" in result:
            response = result["response"]

            # Kiểm tra response có hợp lệ
            if response and not any(msg in str(response)
                                    for msg in ["No information found", "Error"]):
                # Format response dựa vào kiểu dữ liệu
                if isinstance(response, list):
                    return "\n".join(str(item) for item in response)
                else:
                    return str(response)

            # Fallback to query if response is invalid
            if "query" in result and result["query"]:
                return str(result["query"])

            # Return whatever response we have
            return str(response) if response else "No information found"

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
