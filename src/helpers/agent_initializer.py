"""Lazy initialization and one shared execution path for every interface."""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from src.config.settings import Config
from src.handlers.conversation_handler import get_memory_store
from src.handlers.grounding_verifier import select_controlled_agent_response
from src.handlers.memory_manager import ConversationBufferMemory
from src.helpers.checkpoint_utils import (
    checkpoint_thread_id,
    current_turn_prompt,
)
from src.helpers.llm_initializer import get_llm
from src.helpers.logging_config import logger
from src.helpers.prompts import get_healthcare_system_prompt
from src.helpers.security_context import doctor_security_context
from src.helpers.tools import (
    get_last_query,
    medical_guideline_tool,
    patient_guideline_tool,
    rag_tool,
    set_last_query,
)


@dataclass(frozen=True)
class AgentTurnResult:
    """Controlled response plus operational metadata for one completed turn."""

    response: str
    query: Any
    memory_saved: bool


class AgentInitializer:
    """Singleton that owns the LLM, tools and PostgreSQL checkpointer."""

    _instance = None
    _ready = False
    _initialize_lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._ready = False
            cls._instance._initialize_lock = Lock()
        return cls._instance

    def __init__(self) -> None:
        self.config = Config()

    def _ensure_initialized(self) -> None:
        """Connect lazily after interface-level configuration validation."""
        if self._ready:
            return
        with self._initialize_lock:
            if self._ready:
                return
            self.store = get_memory_store()
            self.memory = self.store.checkpointer
            self.llm = get_llm()
            self.tools = [
                rag_tool,
                medical_guideline_tool,
                patient_guideline_tool,
            ]
            self.agent = create_react_agent(
                self.llm,
                self.tools,
                checkpointer=self.memory,
                prompt=current_turn_prompt,
            )
            self._ready = True

    def get_agent(self):
        """Return the initialized LangGraph agent."""
        self._ensure_initialized()
        return self.agent

    def get_memory(
        self, thread_id: str, doctor_id: str
    ) -> ConversationBufferMemory:
        """Return a fresh facade; reads always reflect current PostgreSQL data."""
        self._ensure_initialized()
        return ConversationBufferMemory(
            thread_id,
            doctor_id,
            store=self.store,
            config=self.config,
        )

    def get_conversation_context(self, thread_id: str, doctor_id: str) -> str:
        """Return bounded summary plus recent turns for prompt construction."""
        if not thread_id:
            return ""
        return self.get_memory(
            thread_id, doctor_id
        ).get_conversation_context()["prompt_context"]

    @staticmethod
    def checkpoint_thread_id(thread_id: str, doctor_id: str) -> str:
        """Scope checkpoint identity to both doctor and conversation."""
        return checkpoint_thread_id(thread_id, doctor_id)

    def run_turn(
        self, thread_id: str, doctor_id: str, question: str
    ) -> AgentTurnResult:
        """Execute, ground and persist one turn consistently across interfaces."""
        self._ensure_initialized()
        context = self.get_conversation_context(thread_id, doctor_id)
        system_message = SystemMessage(
            content=get_healthcare_system_prompt(context)
        )
        config = {
            "configurable": {
                "thread_id": self.checkpoint_thread_id(thread_id, doctor_id)
            }
        }

        with doctor_security_context(doctor_id, question):
            set_last_query(None)
            full_response = self.agent.invoke(
                {
                    "messages": [
                        system_message,
                        HumanMessage(content=question),
                    ]
                },
                config,
            )
            response = select_controlled_agent_response(full_response)
            query = get_last_query()

        memory_saved = True
        try:
            self.store.store_turn(thread_id, doctor_id, question, response)
        except Exception:  # pylint: disable=broad-exception-caught
            memory_saved = False
            logger.exception(
                "Response completed but conversation persistence failed "
                "thread_id=%s doctor_id=%s",
                thread_id,
                doctor_id,
            )
        finally:
            self._clear_completed_turn(config)

        return AgentTurnResult(
            response=response,
            query=query,
            memory_saved=memory_saved,
        )

    def _clear_completed_turn(self, config: Dict[str, Any]) -> None:
        """Delete operational state after the controlled turn has completed."""
        try:
            self.store.delete_checkpoint_thread(
                config["configurable"]["thread_id"]
            )
        except Exception:  # pylint: disable=broad-exception-caught
            # The answer is already controlled and persisted. A cleanup failure
            # is operationally important but must not rewrite that answer.
            logger.exception(
                "Unable to delete completed LangGraph checkpoint thread_id=%s",
                config["configurable"]["thread_id"],
            )

    def compact_conversation(self, thread_id: str, doctor_id: str) -> bool:
        """Run rolling-summary compaction for one doctor-scoped thread."""
        self._ensure_initialized()
        return self.get_memory(thread_id, doctor_id).compact(self.llm)

    def delete_thread_memory(self, thread_id: str, doctor_id: str) -> bool:
        """Delete both explicit chat memory and the scoped agent checkpoint."""
        self._ensure_initialized()
        self.store.delete_thread(thread_id, doctor_id)
        self.store.delete_checkpoint_thread(
            self.checkpoint_thread_id(thread_id, doctor_id)
        )
        return True


agent_initializer = AgentInitializer()
