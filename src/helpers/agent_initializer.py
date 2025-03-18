from src.config.settings import Config
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from src.helpers.tools import rag_tool, llm_tool
from src.helpers.llm_initializer import get_llm


class AgentInitializer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgentInitializer, cls).__new__(cls)
            cls._instance._initialize_agent()
        return cls._instance

    def _initialize_agent(self):
        config = Config()
        memory = MemorySaver()
        self.llm = get_llm()
        tools = [rag_tool, llm_tool]
        self.agent = create_react_agent(self.llm, tools, checkpointer=memory)

    def get_agent(self):
        return self.agent


agent_initializer = AgentInitializer()
agent_initializer._initialize_agent()
