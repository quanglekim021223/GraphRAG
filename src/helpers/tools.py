from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.helpers.logging_config import logger
from src.handlers.graphrag_handler import HealthcareGraphRAG
from src.config.settings import Config


@tool
def rag_tool(question: str) -> str:
    """Use this tool to query specific healthcare data from the database, such as patient information, diseases, doctor, hospital, insurance provider, room or treatments."""
    try:
        config = Config()
        graphrag = HealthcareGraphRAG(config)
        result = graphrag.run(question)
        logger.info(f"Raw GraphRAG result: {result}")
        if isinstance(result, dict) and "response" in result:
            response = result["response"]
            if response and "No information found" not in response and "Error" not in response:
                if isinstance(response, list):
                    return "\n".join(str(item) for item in response) if response else "No data available"
                elif isinstance(response, str):
                    return response
                else:
                    return str(response)
            if "query" in result and result["query"]:
                return "\n".join(str(item) for item in result["query"]) if result["query"] else "No data available"
            return response if response else "No information found"
        return "No information found"
    except Exception as e:
        logger.error(f"GraphRAG error: {str(e)}")
        return f"Error: {str(e)}"


@tool
def llm_tool(question: str) -> str:
    """Use this tool to provide general medical knowledge or when specific data is not available."""
    try:
        config = Config()
        llm = ChatOpenAI(
            base_url=config.endpoint,
            api_key=config.github_token,
            model_name=config.model_name,
            temperature=0.3
        )
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
