"""
CLI Router module for Healthcare GraphRAG system.

This module provides a command-line interface for interacting with the Healthcare GraphRAG system,
allowing users to ask questions and receive responses in a terminal environment.
It maintains conversation history within a session using a thread ID.
"""
import re
import uuid

from langchain_core.messages import SystemMessage, HumanMessage

from src.config.settings import Config
from src.helpers.agent_initializer import agent_initializer
from src.helpers.logging_config import logger
from src.helpers.tools import get_last_query, set_last_query
from src.helpers.prompts import get_healthcare_system_prompt
from src.handlers.grounding_verifier import select_controlled_agent_response
from src.helpers.security_context import doctor_security_context
from src.handlers.security_guardrails import check_prompt_injection


def run_cli():
    """Run the application in CLI mode."""
    config = Config()
    config.validate()

    # Initialize ReAct agent with both tools
    agent_executor = agent_initializer.get_agent()

    print("Healthcare GraphRAG CLI")
    print("Type 'exit' to quit")

    # Generate a permanent thread_id for this CLI session
    session_thread_id = str(uuid.uuid4())
    print(f"Session ID: {session_thread_id}")
    doctor_id = input("Authenticated doctor ID: ").strip()
    if not doctor_id:
        print("Doctor ID is required.")
        return

    while True:
        question = input("\nEnter your question: ")
        if question.lower() in ['exit', 'quit', 'q']:
            break
        if check_prompt_injection(question):
            print(
                "Yêu cầu này cần được kiểm tra thủ công trước khi truy cập dữ liệu."
            )
            continue

        try:
            # Sử dụng prompt chung từ helpers/prompts.py
            system_content = get_healthcare_system_prompt()
            system_message = SystemMessage(content=system_content)

            # Provide thread_id in configurable
            config_obj = {"configurable": {
                "thread_id": f"{doctor_id}:{session_thread_id}"
            }}

            # Invoke agent and get full response
            with doctor_security_context(doctor_id):
                set_last_query(None)
                full_response = agent_executor.invoke(
                    {"messages": [system_message, HumanMessage(content=question)]},
                    config_obj
                )

            # Bypass any ReAct paraphrase when grounded database data was used.
            agent_response = select_controlled_agent_response(full_response)
            print(f"\n🔍 Response: {agent_response}")

            # Try to display query info if available
            query_info = get_last_query()
            if query_info:
                print(f"\n📊 Cypher Query: {query_info}")

        except Exception as e:  # pylint: disable=broad-exception-caught
            # Broad exception catch is necessary here as CLI entry point
            logger.error("Error processing question: %s", str(e))
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    run_cli()
