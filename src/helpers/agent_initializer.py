from src.config.settings import Config
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from src.helpers.tools import rag_tool, llm_tool
from src.helpers.llm_initializer import get_llm
from src.handlers.memory_manager import ConversationBufferMemory
from langchain_core.messages import HumanMessage
from typing import List


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
        self.memory_manager = {}  # Thread ID -> ConversationBufferMemory

    def get_agent(self):
        return self.agent

    def get_memory(self, thread_id: str) -> ConversationBufferMemory:
        """Get or create memory for thread."""
        if thread_id not in self.memory_manager:
            self.memory_manager[thread_id] = ConversationBufferMemory(
                thread_id)
        return self.memory_manager[thread_id]

    def get_conversation_context(self, thread_id: str) -> str:
        """Get full conversation context for the given thread."""
        if not thread_id:
            return ""

        memory = self.get_memory(thread_id)
        context = memory.get_conversation_context()

        context_parts = []

        # Add full conversation context with proper formatting
        if "conversation" in context and context["conversation"]:
            context_parts.append("## Previous conversation:")
            # Limit to a reasonable length if needed
            convo_text = context["conversation"]
            if len(convo_text.split("\n")) > 10:
                # If conversation is long, append only last 10 turns
                convo_lines = convo_text.split("\n")
                context_parts.append(
                    "(Showing most recent conversation turns)")
                context_parts.append("\n".join(convo_lines[-10:]))
            else:
                context_parts.append(convo_text)

        # Add key topics if available
        if "topics" in context and context["topics"]:
            context_parts.append("\n## Key information mentioned:")
            for topic in context["topics"]:
                context_parts.append(f"- {topic}")

        return "\n".join(context_parts)


agent_initializer = AgentInitializer()
agent_initializer._initialize_agent()
