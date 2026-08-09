"""Pure helpers for bounded, doctor-scoped LangGraph checkpoint state."""
from typing import Any, Dict, List

from langchain_core.messages import SystemMessage


def current_turn_prompt(state: Dict[str, Any]) -> List[Any]:
    """Return only the latest request segment from accumulated graph state."""
    messages = list(state.get("messages", []))
    latest_system_index = 0
    for index, message in enumerate(messages):
        if isinstance(message, SystemMessage):
            latest_system_index = index
    return messages[latest_system_index:]


def checkpoint_thread_id(thread_id: str, doctor_id: str) -> str:
    """Build a tenant-scoped LangGraph thread key."""
    return f"{doctor_id}:{thread_id}"
