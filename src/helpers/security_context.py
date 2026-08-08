"""Trusted request values kept outside LLM-controlled tool arguments."""
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional


_CURRENT_DOCTOR_ID: ContextVar[Optional[str]] = ContextVar(
    "current_doctor_id", default=None
)
_CURRENT_USER_QUESTION: ContextVar[Optional[str]] = ContextVar(
    "current_user_question", default=None
)
_CURRENT_TOOL_NAME: ContextVar[Optional[str]] = ContextVar(
    "current_tool_name", default=None
)


def get_current_doctor_id() -> str:
    """Return request identity or fail closed when the boundary did not set it."""
    doctor_id = _CURRENT_DOCTOR_ID.get()
    if not doctor_id:
        raise ValueError("Authenticated doctor identity is missing")
    return doctor_id


def get_current_user_question() -> str:
    """Return the immutable current-turn question or fail closed."""
    question = _CURRENT_USER_QUESTION.get()
    if not question:
        raise ValueError("Current user question is missing")
    return question


def claim_request_tool(tool_name: str) -> bool:
    """Allow exactly one outer data-bearing tool invocation per request."""
    if _CURRENT_TOOL_NAME.get() is not None:
        return False
    _CURRENT_TOOL_NAME.set(tool_name)
    return True


@contextmanager
def doctor_security_context(
    doctor_id: str, question: Optional[str] = None
) -> Iterator[None]:
    """Bind trusted request values so ReAct cannot choose or modify them."""
    if not isinstance(doctor_id, str) or not doctor_id.strip():
        raise ValueError("Authenticated doctor identity is missing")
    doctor_token = _CURRENT_DOCTOR_ID.set(doctor_id.strip())
    question_token = _CURRENT_USER_QUESTION.set(
        question.strip() if isinstance(question, str) and question.strip() else None
    )
    tool_token = _CURRENT_TOOL_NAME.set(None)
    try:
        yield
    finally:
        _CURRENT_TOOL_NAME.reset(tool_token)
        _CURRENT_USER_QUESTION.reset(question_token)
        _CURRENT_DOCTOR_ID.reset(doctor_token)
