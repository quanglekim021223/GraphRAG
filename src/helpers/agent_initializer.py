"""
Agent Initializer module for Healthcare GraphRAG system.

This module provides Singleton implementation for the agent system,
manages memory contexts across conversation threads, and initializes
the required tools and language models for the AI assistant.
"""
from typing import Dict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from src.config.settings import Config
from src.handlers.memory_manager import ConversationBufferMemory
from src.helpers.llm_initializer import get_llm
from src.helpers.tools import rag_tool, llm_tool


class AgentInitializer:
    """
    Singleton class that initializes and manages the agent and its memory.

    This class is responsible for creating the agent with appropriate tools,
    managing conversation memory across threads, and providing conversation
    context for prompts.
    """

    _instance = None

    def __new__(cls):
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super(AgentInitializer, cls).__new__(cls)
            # Khởi tạo flag trong __new__ để tránh lỗi access before definition
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize agent and its components if not already initialized."""
        # Sử dụng _initialized (single underscore) thay vì __initialized
        if not self._initialized:
            # Initialize agent components
            self.config = Config()
            self.memory = MemorySaver()
            self.llm = get_llm()
            self.tools = [rag_tool, llm_tool]
            self.agent = create_react_agent(
                self.llm, self.tools, checkpointer=self.memory)
            self.memory_manager: Dict[str, ConversationBufferMemory] = {}
            self._initialized = True

    def get_agent(self):
        """
        Get the initialized agent instance.

        Returns:
            The LangGraph agent instance ready for invocation
        """
        return self.agent

    def get_memory(self, thread_id: str) -> ConversationBufferMemory:
        """
        Get or create memory for thread.

        Args:
            thread_id: The unique identifier for the conversation thread

        Returns:
            ConversationBufferMemory: Memory instance for the specified thread
        """
        if thread_id not in self.memory_manager:
            self.memory_manager[thread_id] = ConversationBufferMemory(
                thread_id)
        return self.memory_manager[thread_id]

    def get_conversation_context(self, thread_id: str) -> str:
        """
        Get full conversation context for the given thread.

        Args:
            thread_id: The unique identifier for the conversation thread

        Returns:
            String representation of the conversation context for prompts
        """
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


# Create the singleton instance
agent_initializer = AgentInitializer()
