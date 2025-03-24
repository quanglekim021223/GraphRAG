"""
API Router module for Healthcare GraphRAG system.

This module provides FastAPI endpoints for interacting with the Healthcare GraphRAG system,
including chat functionality and persistent conversation history across sessions.
"""
import uuid
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from src.helpers.agent_initializer import agent_initializer
from src.config.settings import Config
from src.helpers.logging_config import logger
from src.helpers.tools import get_last_query
from src.helpers.prompts import get_healthcare_system_prompt


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
    agent_executor = agent_initializer.get_agent()

    @app.get("/")
    def home():
        return {"message": "Welcome to the Healthcare GraphRAG chatbot (FastAPI)!"}

    @app.post("/chat")
    async def chat(request_body: ChatRequest):
        question = request_body.question
        thread_id = request_body.thread_id or str(uuid.uuid4())

        if not question:
            raise HTTPException(status_code=400, detail="Question is required")

        try:
            # Sử dụng prompt chung từ helpers/prompts.py
            system_content = get_healthcare_system_prompt()
            system_message = SystemMessage(content=system_content)

            # Truyền thread_id vào config
            config_obj = {"configurable": {"thread_id": thread_id}}

            # Gọi agent
            full_response = await agent_executor.ainvoke(
                {"messages": [system_message, HumanMessage(content=question)]},
                config_obj
            )

            # Lấy content trả về
            agent_response = full_response["messages"][-1].content

            # Trích xuất Cypher query từ rag_tool nếu có
            query_info = get_last_query()

            return {
                "question": question,
                "response": agent_response,
                "query": query_info,
                "thread_id": thread_id
            }

        except Exception as e:
            logger.error("API error: %s", str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


def run_api(port=5000):
    """Hàm chạy server FastAPI bằng uvicorn, thay vì Flask."""
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=port)
