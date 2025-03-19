# src/routers/api_router.py

import re
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Optional
from langchain_core.messages import SystemMessage, HumanMessage
from src.helpers.agent_initializer import agent_initializer
from src.config.settings import Config
from src.helpers.logging_config import logger
from src.helpers.tools import get_last_query
# Bổ sung class mô tả payload request


class ChatRequest(BaseModel):
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
    async def home():
        return {"message": "Welcome to the Healthcare GraphRAG chatbot (FastAPI)!"}

    @app.post("/chat")
    async def chat(request_body: ChatRequest):
        question = request_body.question
        thread_id = request_body.thread_id or str(uuid.uuid4())

        if not question:
            raise HTTPException(status_code=400, detail="Question is required")

        try:
            # SystemMessage
            system_message = SystemMessage(content="""
                You are a healthcare assistant. Based on the user's question:
                - If the question is about specific patient data, diseases, doctor, hospital, insurance provider, room or treatments, use the 'rag_tool'.
                - If the question is general or no specific data is needed, use the 'llm_tool'.
            """)

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
            logger.error(f"API error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    return app


def run_api(port=5000):
    """Hàm chạy server FastAPI bằng uvicorn, thay vì Flask."""
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=port)
