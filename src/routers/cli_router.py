"""
CLI Router module for Healthcare GraphRAG system.

This module provides a command-line interface for interacting with the Healthcare GraphRAG system,
allowing users to ask questions and receive responses in a terminal environment.
It maintains conversation history within a session using a thread ID.
"""
import uuid

from src.config.settings import Config
from src.helpers.agent_initializer import agent_initializer
from src.helpers.logging_config import logger
from src.handlers.security_guardrails import check_prompt_injection


def run_cli():
    """Run the application in CLI mode."""
    config = Config()
    config.validate()

    # Initialize ReAct agent with grounded lookup and deterministic refusal.
    agent_initializer.get_agent()

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
            result = agent_initializer.run_turn(
                session_thread_id, doctor_id, question
            )
            print(f"\n🔍 Response: {result.response}")
            if not result.memory_saved:
                print("⚠️ Memory was not persisted for this turn.")

            # Try to display query info if available
            if result.query:
                print(f"\n📊 Cypher Query: {result.query}")

            # Usually a no-op; invokes the summarizer only after the threshold.
            if result.memory_saved:
                agent_initializer.compact_conversation(
                    session_thread_id, doctor_id
                )

        except Exception as e:  # pylint: disable=broad-exception-caught
            # Broad exception catch is necessary here as CLI entry point
            logger.error("Error processing question: %s", str(e))
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    run_cli()
