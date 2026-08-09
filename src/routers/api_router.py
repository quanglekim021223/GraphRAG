"""
API Router module for Healthcare GraphRAG system.

This module provides FastAPI endpoints for interacting with the Healthcare GraphRAG system,
including chat functionality and persistent conversation history across sessions.
"""
import asyncio
import uuid
from typing import Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from pydantic import BaseModel
from src.helpers.agent_initializer import agent_initializer
from src.config.settings import Config
from src.helpers.logging_config import logger
from src.handlers.security_guardrails import check_prompt_injection


class ChatRequest(BaseModel):
    """
    Request payload model for chat interactions.

    Contains the user's question and an optional thread ID for conversation continuity.
    If thread_id is not provided, a new one will be generated.
    """
    question: str
    thread_id: Optional[str] = None


def create_app():
    """Tạo app FastAPI, khởi tạo agent và đăng ký các endpoint."""
    app = FastAPI(title="Healthcare GraphRAG API")

    # Load config
    config = Config()
    config.validate()

    # Khởi tạo ReAct agent
    agent_initializer.get_agent()

    @app.get("/")
    def home():
        return {"message": "Welcome to the Healthcare GraphRAG chatbot (FastAPI)!"}

    @app.post("/chat")
    async def chat(
        request_body: ChatRequest,
        background_tasks: BackgroundTasks,
        doctor_id: str = Header(..., alias="X-Doctor-ID"),
    ):
        question = request_body.question
        thread_id = request_body.thread_id or str(uuid.uuid4())

        if not question:
            raise HTTPException(status_code=400, detail="Question is required")
        if check_prompt_injection(question):
            return {
                "question": question,
                "response": (
                    "Yêu cầu này cần được kiểm tra thủ công trước khi truy cập dữ liệu."
                ),
                "query": None,
                "thread_id": thread_id,
                "status": "manual_review",
            }

        try:
            # The sync saver is called in a worker so the event loop remains
            # responsive and request-local ContextVars stay inside one thread.
            result = await asyncio.to_thread(
                agent_initializer.run_turn,
                thread_id,
                doctor_id,
                question,
            )
            if result.memory_saved:
                background_tasks.add_task(
                    agent_initializer.compact_conversation,
                    thread_id,
                    doctor_id,
                )

            return {
                "question": question,
                "response": result.response,
                "query": result.query,
                "thread_id": thread_id,
                "memory_persisted": result.memory_saved,
            }

        except Exception as e:
            logger.error("API error: %s", str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


def run_api(port=5000):
    """Hàm chạy server FastAPI bằng uvicorn, thay vì Flask."""
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=port)
