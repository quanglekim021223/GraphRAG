"""
Memory Manager module for Healthcare GraphRAG system.

This module implements conversation memory components using Neo4j as a persistent
storage backend. It provides classes for history management, topic extraction,
and conversation context formatting for the AI agent system.
"""
from typing import Dict, List, Any, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.chat_history import BaseChatMessageHistory
from src.handlers.conversation_handler import get_conversation_history


class Neo4jChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in Neo4j database."""

    def __init__(self, thread_id: str):
        """Initialize with thread ID."""
        self.thread_id = thread_id
        self.messages = []
        self._load_from_neo4j()

    def _load_from_neo4j(self):
        """Load messages from Neo4j."""
        if not self.thread_id:
            return

        history = get_conversation_history(self.thread_id)
        messages = []
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        self.messages = messages

    def add_message(self, message: Any) -> None:
        """Add a message to the history."""
        # Note: We don't need to store here as this is handled by store_conversation
        self.messages.append(message)

    def clear(self) -> None:
        """Clear message history."""
        self.messages = []

    def get_messages(self) -> List[Any]:
        """Get messages."""
        return self.messages


class ConversationBufferMemory:
    """Memory manager that uses Neo4j for persistent storage."""

    def __init__(self, thread_id: Optional[str] = None):
        self.thread_id = thread_id
        self.chat_memory = Neo4jChatMessageHistory(
            thread_id) if thread_id else None

    def set_thread_id(self, thread_id: str):
        """Update thread ID and reload memory."""
        self.thread_id = thread_id
        self.chat_memory = Neo4jChatMessageHistory(thread_id)

    def get_chat_history(self) -> str:
        """Format chat history as a string for context."""
        if not self.chat_memory:
            return ""

        formatted_messages = []
        for msg in self.chat_memory.messages:
            if isinstance(msg, HumanMessage):
                formatted_messages.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted_messages.append(f"Assistant: {msg.content}")

        return "\n".join(formatted_messages)

    def get_conversation_context(self) -> Dict[str, Any]:
        """Get comprehensive context from conversation history."""
        if not self.chat_memory:
            return {}

        # Format conversation
        conversation = self.get_chat_history()

        # Extract key topics/entities mentioned in conversation
        topics = self._extract_conversation_topics()

        return {
            "conversation": conversation,
            "topics": topics,
        }

    def _extract_conversation_topics(self) -> List[str]:
        """Extract key topics from the conversation history."""
        if not self.chat_memory:
            return []

        topics = set()

        for msg in self.chat_memory.messages:
            if isinstance(msg, HumanMessage):
                # Extract basic topics based on common healthcare terms
                content = msg.content.lower()

                # Add healthcare keywords mentioned
                healthcare_keywords = ["patient", "doctor", "hospital", "disease",
                                       "treatment", "medication", "diagnosis",
                                       "symptoms", "insurance", "appointment"]

                for keyword in healthcare_keywords:
                    if keyword in content:
                        topics.add(keyword)

                # Extract potential names and personal info
                if "my name is" in content:
                    name_part = content.split("my name is")[1].strip()
                    name = name_part.split()[0].strip(".,!?")
                    topics.add(f"user's name: {name}")

                if "i am" in content and ("years old" in content or "year old" in content):
                    try:
                        age_text = content.split(
                            "i am")[1].split("year")[0].strip()
                        age_num = ''.join(filter(str.isdigit, age_text))
                        if age_num:
                            topics.add(f"user's age: {age_num}")
                    except Exception:  # Thay bare except bằng Exception
                        pass

        return list(topics)
