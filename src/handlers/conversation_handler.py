"""
Conversation handler module for Healthcare GraphRAG.

This module provides functions to store, retrieve, and manage conversation history in Neo4j,
including thread management, message storage and retrieval for memory contexts.
"""
from datetime import datetime
from typing import List, Dict
from src.helpers.logging_config import logger
from src.handlers.graphrag_handler import HealthcareGraphRAG
from src.config.settings import Config

# Khởi tạo singleton instances ở cấp module
config = Config()
graphrag = HealthcareGraphRAG()


def store_conversation(thread_id: str, user_input: str, response: str):
    """Store conversation history in Neo4j."""
    try:
        # pylint: disable=protected-access
        with graphrag.graph_manager.graph._driver.session() as session:
            session.run(
                """
                MERGE (c:Conversation {thread_id: $thread_id})
                CREATE (m:Message {user_input: $user_input, response: $response, timestamp: $timestamp})
                CREATE (c)-[:HAS_MESSAGE]->(m)
                """,
                {"thread_id": thread_id,
                 "user_input": user_input,
                 "response": response,
                 "timestamp": str(datetime.now())}
            )
        logger.info("Stored conversation for thread_id: %s", thread_id)
    except Exception as e:  # pylint: disable=broad-exception-caught
        # Bắt tất cả exception vì đây là hàm wrapper cần xử lý mọi lỗi database
        logger.error("Error storing conversation: %s", str(e))


def get_conversation_history(thread_id: str) -> List[Dict[str, str]]:
    """Retrieve conversation history from Neo4j."""
    try:
        # pylint: disable=protected-access
        with graphrag.graph_manager.graph._driver.session() as session:
            result = session.run(
                """
                MATCH (c:Conversation {thread_id: $thread_id})-[:HAS_MESSAGE]->(m:Message)
                RETURN m.user_input AS user_input, m.response AS response, m.timestamp AS timestamp
                ORDER BY m.timestamp
                """,
                {"thread_id": thread_id}
            )
            history = []
            for record in result:
                history.append(
                    {"role": "user", "content": record["user_input"]})
                history.append(
                    {"role": "assistant", "content": record["response"]})
            return history
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error retrieving conversation: %s", str(e))
        return []


def get_all_conversations() -> List[str]:
    """Retrieve all conversation thread_ids from Neo4j."""
    try:
        # pylint: disable=protected-access
        with graphrag.graph_manager.graph._driver.session() as session:
            result = session.run(
                """
                MATCH (c:Conversation)
                RETURN c.thread_id AS thread_id
                """
            )
            return [record["thread_id"] for record in result]
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error retrieving conversations: %s", str(e))
        return []


def delete_conversation(thread_id: str) -> bool:
    """Delete a conversation and its messages from Neo4j."""
    try:
        # pylint: disable=protected-access
        with graphrag.graph_manager.graph._driver.session() as session:
            session.run(
                """
                MATCH (c:Conversation {thread_id: $thread_id})-[:HAS_MESSAGE]->(m:Message)
                DETACH DELETE c, m
                """,
                {"thread_id": thread_id}
            )
        logger.info("Deleted conversation for thread_id: %s", thread_id)
        return True
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Error deleting conversation: %s", str(e))
        return False
