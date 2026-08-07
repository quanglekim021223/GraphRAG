"""Request-local authenticated doctor identity, kept outside LLM tool arguments."""
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional


_CURRENT_DOCTOR_ID: ContextVar[Optional[str]] = ContextVar(
    "current_doctor_id", default=None
)


def get_current_doctor_id() -> str:
    """Return request identity or fail closed when the boundary did not set it."""
    doctor_id = _CURRENT_DOCTOR_ID.get()
    if not doctor_id:
        raise ValueError("Authenticated doctor identity is missing")
    return doctor_id


@contextmanager
def doctor_security_context(doctor_id: str) -> Iterator[None]:
    """Bind identity for one request so ReAct cannot choose or modify it."""
    if not isinstance(doctor_id, str) or not doctor_id.strip():
        raise ValueError("Authenticated doctor identity is missing")
    token = _CURRENT_DOCTOR_ID.set(doctor_id.strip())
    try:
        yield
    finally:
        _CURRENT_DOCTOR_ID.reset(token)
