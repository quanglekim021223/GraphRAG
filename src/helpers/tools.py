from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from src.helpers.logging_config import logger
from src.handlers.graphrag_handler import HealthcareGraphRAG
from src.helpers.llm_initializer import get_llm

graphrag_instance = HealthcareGraphRAG()
llm = get_llm()

# Thêm biến toàn cục ở đầu file
_last_query = None


def get_last_query():
    global _last_query
    return _last_query


def set_last_query(query):
    global _last_query
    _last_query = query


@tool
def rag_tool(question: str) -> str:
    """Use this tool to query specific healthcare data from the database."""
    try:
        result = graphrag_instance.run(question)
        logger.info(f"Raw GraphRAG result: {result}")

        # Lưu query nếu có
        if isinstance(result, dict) and "query" in result:
            set_last_query(result["query"])

        # Xử lý response
        if isinstance(result, dict) and "response" in result:
            response = result["response"]

            # Kiểm tra response có hợp lệ
            if response and not any(msg in str(response) for msg in ["No information found", "Error"]):
                # Format response dựa vào kiểu dữ liệu
                return "\n".join(str(item) for item in response) if isinstance(response, list) else str(response)

            # Fallback to query if response is invalid
            if "query" in result and result["query"]:
                return str(result["query"])

            # Return whatever response we have
            return str(response) if response else "No information found"

        return "No information found"
    except Exception as e:
        logger.error(f"GraphRAG error: {str(e)}")
        return f"Error: {str(e)}"


@tool
def llm_tool(question: str) -> str:
    """Use this tool to provide general medical knowledge or when specific data is not available."""
    try:

        general_prompt = (
            f"You are a healthcare assistant. User asked: '{question}'. No specific data was found in the database. "
            "Provide a general answer based on common medical knowledge and append a follow-up question "
            "like 'Do you have any more questions?' or 'Is this the information you were looking for?'"
        )
        return llm.invoke([
            SystemMessage(content="You are a healthcare assistant."),
            HumanMessage(content=general_prompt)
        ]).content.strip()
    except Exception as e:
        logger.error(f"LLM tool error: {str(e)}")
        return f"Error generating response: {str(e)}"
