from src.config.settings import Config
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from src.helpers.tools import rag_tool, llm_tool


def initialize_agent():
    """Initialize the ReAct agent with tools.

    Returns:
        ReAct agent ready to process queries using RAG or LLM tools.
    """
    config = Config()
    memory = MemorySaver()
    model = ChatOpenAI(
        base_url=config.endpoint,
        api_key=config.github_token,
        model_name=config.model_name,
        temperature=0.3
    )
    tools = [rag_tool, llm_tool]
    return create_react_agent(model, tools, checkpointer=memory)
