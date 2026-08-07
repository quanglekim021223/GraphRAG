"""Request-local authenticated doctor identity, kept outside LLM tool arguments."""
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional


_CURRENT_DOCTOR_ID: ContextVar[Optional[str]] = ContextVar(
    "current_doctor_id", default=None
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


def claim_request_tool(tool_name: str) -> bool:
    """Allow exactly one data-bearing tool invocation per request."""
    if _CURRENT_TOOL_NAME.get() is not None:
        return False
    _CURRENT_TOOL_NAME.set(tool_name)
    return True


@contextmanager
def doctor_security_context(doctor_id: str) -> Iterator[None]:
    """Bind identity for one request so ReAct cannot choose or modify it."""
    if not isinstance(doctor_id, str) or not doctor_id.strip():
        raise ValueError("Authenticated doctor identity is missing")
    doctor_token = _CURRENT_DOCTOR_ID.set(doctor_id.strip())
    tool_token = _CURRENT_TOOL_NAME.set(None)
    try:
        yield
    finally:
        _CURRENT_TOOL_NAME.reset(tool_token)
        _CURRENT_DOCTOR_ID.reset(doctor_token)
