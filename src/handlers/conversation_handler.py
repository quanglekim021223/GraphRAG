"""Doctor-scoped conversation persistence backed by PostgreSQL.

Neo4j remains the source of truth for patient records. PostgreSQL stores the
application's operational state: LangGraph checkpoints, raw conversation turns
and a rolling summary. The summary is navigation context only; medical claims
must always be retrieved again from Neo4j or the trusted guideline corpus.

The checkpointer is operational, not conversational: checkpoints exist while a
graph turn is running and are deleted after the controlled response is stored.
The explicit conversation tables provide durable history and auditability.
"""
from __future__ import annotations

import atexit
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Optional

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from src.config.settings import Config
from src.helpers.logging_config import logger


@dataclass(frozen=True)
class CompactionBatch:
    """Immutable snapshot used to update one rolling summary safely."""

    previous_summary: str
    expected_through_message_id: int
    new_through_message_id: int
    turns: List[Dict[str, Any]]


class PostgresConversationStore:
    """Own one shared connection pool for checkpoints and chat memory."""

    def __init__(
        self,
        postgres_uri: str,
        pool: Optional[ConnectionPool] = None,
    ) -> None:
        self._owns_pool = pool is None
        self.pool = pool or ConnectionPool(
            conninfo=postgres_uri,
            min_size=1,
            max_size=10,
            timeout=10,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
            open=True,
        )
        if self._owns_pool:
            self.pool.wait(timeout=10)

        self.checkpointer = PostgresSaver(self.pool)
        self.checkpointer.setup()
        self._setup_conversation_schema()

        if self._owns_pool:
            atexit.register(self.close)

    def _setup_conversation_schema(self) -> None:
        """Create the small application-owned schema idempotently."""
        statements = (
            """
            CREATE TABLE IF NOT EXISTS conversation_threads (
                doctor_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                summary_through_message_id BIGINT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                summary_updated_at TIMESTAMPTZ,
                PRIMARY KEY (doctor_id, thread_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS conversation_messages (
                message_id BIGSERIAL PRIMARY KEY,
                doctor_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                user_input TEXT NOT NULL,
                response TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                FOREIGN KEY (doctor_id, thread_id)
                    REFERENCES conversation_threads (doctor_id, thread_id)
                    ON DELETE CASCADE
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_conversation_messages_thread
            ON conversation_messages (doctor_id, thread_id, message_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_conversation_threads_updated
            ON conversation_threads (doctor_id, updated_at DESC)
            """,
        )
        with self.pool.connection() as connection:
            for statement in statements:
                connection.execute(statement)

    def close(self) -> None:
        """Close only pools created by this store."""
        if self._owns_pool and not self.pool.closed:
            self.pool.close()

    def store_turn(
        self,
        thread_id: str,
        doctor_id: str,
        user_input: str,
        response: str,
    ) -> int:
        """Persist one user/assistant pair atomically and return its row ID."""
        with self.pool.connection() as connection:
            with connection.transaction():
                connection.execute(
                    """
                    INSERT INTO conversation_threads (doctor_id, thread_id)
                    VALUES (%s, %s)
                    ON CONFLICT (doctor_id, thread_id)
                    DO UPDATE SET updated_at = NOW()
                    """,
                    (doctor_id, thread_id),
                )
                row = connection.execute(
                    """
                    INSERT INTO conversation_messages (
                        doctor_id, thread_id, user_input, response
                    )
                    VALUES (%s, %s, %s, %s)
                    RETURNING message_id
                    """,
                    (doctor_id, thread_id, user_input, response),
                ).fetchone()

        message_id = int(row["message_id"])
        logger.info(
            "Stored PostgreSQL conversation turn thread_id=%s doctor_id=%s",
            thread_id,
            doctor_id,
        )
        return message_id

    def get_history(
        self, thread_id: str, doctor_id: str
    ) -> List[Dict[str, str]]:
        """Return the full UI/audit history as alternating chat messages."""
        with self.pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT user_input, response
                FROM conversation_messages
                WHERE doctor_id = %s AND thread_id = %s
                ORDER BY message_id
                """,
                (doctor_id, thread_id),
            ).fetchall()

        history: List[Dict[str, str]] = []
        for row in rows:
            history.append({"role": "user", "content": row["user_input"]})
            history.append({"role": "assistant", "content": row["response"]})
        return history

    def get_context_snapshot(
        self,
        thread_id: str,
        doctor_id: str,
        recent_turns: int,
        summary_trigger_turns: int,
    ) -> Dict[str, Any]:
        """Return summary plus a bounded set of not-yet-summarized turns."""
        with self.pool.connection() as connection:
            thread = connection.execute(
                """
                SELECT summary, COALESCE(summary_through_message_id, 0)
                    AS summary_through_message_id
                FROM conversation_threads
                WHERE doctor_id = %s AND thread_id = %s
                """,
                (doctor_id, thread_id),
            ).fetchone()
            if thread is None:
                return {"summary": "", "turns": []}

            turns = connection.execute(
                """
                SELECT message_id, user_input, response
                FROM conversation_messages
                WHERE doctor_id = %s AND thread_id = %s
                  AND message_id > %s
                ORDER BY message_id
                """,
                (
                    doctor_id,
                    thread_id,
                    thread["summary_through_message_id"],
                ),
            ).fetchall()

        # Before compaction is required, retaining all pending turns prevents a
        # gap between the stored summary and the recent window.
        if len(turns) > summary_trigger_turns:
            turns = turns[-recent_turns:]
        return {"summary": thread["summary"], "turns": list(turns)}

    def get_compaction_batch(
        self,
        thread_id: str,
        doctor_id: str,
        recent_turns: int,
        summary_trigger_turns: int,
    ) -> Optional[CompactionBatch]:
        """Select older pending turns while keeping a recent verbatim window."""
        with self.pool.connection() as connection:
            thread = connection.execute(
                """
                SELECT summary, COALESCE(summary_through_message_id, 0)
                    AS summary_through_message_id
                FROM conversation_threads
                WHERE doctor_id = %s AND thread_id = %s
                """,
                (doctor_id, thread_id),
            ).fetchone()
            if thread is None:
                return None

            turns = connection.execute(
                """
                SELECT message_id, user_input, response
                FROM conversation_messages
                WHERE doctor_id = %s AND thread_id = %s
                  AND message_id > %s
                ORDER BY message_id
                """,
                (
                    doctor_id,
                    thread_id,
                    thread["summary_through_message_id"],
                ),
            ).fetchall()

        if len(turns) <= summary_trigger_turns:
            return None

        turns_to_summarize = list(turns[:-recent_turns])
        return CompactionBatch(
            previous_summary=thread["summary"],
            expected_through_message_id=int(
                thread["summary_through_message_id"]
            ),
            new_through_message_id=int(
                turns_to_summarize[-1]["message_id"]
            ),
            turns=turns_to_summarize,
        )

    def save_summary(
        self,
        thread_id: str,
        doctor_id: str,
        batch: CompactionBatch,
        summary: str,
    ) -> bool:
        """Use optimistic locking so concurrent summarizers cannot overwrite."""
        with self.pool.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE conversation_threads
                SET summary = %s,
                    summary_through_message_id = %s,
                    summary_updated_at = NOW(),
                    updated_at = NOW()
                WHERE doctor_id = %s AND thread_id = %s
                  AND COALESCE(summary_through_message_id, 0) = %s
                """,
                (
                    summary,
                    batch.new_through_message_id,
                    doctor_id,
                    thread_id,
                    batch.expected_through_message_id,
                ),
            )
        return cursor.rowcount == 1

    def purge_expired_summarized_turns(
        self,
        thread_id: str,
        doctor_id: str,
        retention_days: int,
    ) -> int:
        """Delete only expired raw turns already covered by the summary."""
        with self.pool.connection() as connection:
            cursor = connection.execute(
                """
                DELETE FROM conversation_messages AS message
                USING conversation_threads AS thread
                WHERE message.doctor_id = thread.doctor_id
                  AND message.thread_id = thread.thread_id
                  AND thread.doctor_id = %s
                  AND thread.thread_id = %s
                  AND message.message_id <= COALESCE(
                      thread.summary_through_message_id, 0
                  )
                  AND message.created_at < NOW() - (%s * INTERVAL '1 day')
                """,
                (doctor_id, thread_id, retention_days),
            )
        return cursor.rowcount

    def get_thread_ids(self, doctor_id: str) -> List[str]:
        """List only threads owned by the authenticated doctor."""
        with self.pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT thread_id
                FROM conversation_threads
                WHERE doctor_id = %s
                ORDER BY updated_at DESC
                """,
                (doctor_id,),
            ).fetchall()
        return [row["thread_id"] for row in rows]

    def delete_thread(self, thread_id: str, doctor_id: str) -> bool:
        """Delete one doctor's conversation rows; checkpoint is separate."""
        with self.pool.connection() as connection:
            cursor = connection.execute(
                """
                DELETE FROM conversation_threads
                WHERE doctor_id = %s AND thread_id = %s
                """,
                (doctor_id, thread_id),
            )
        return cursor.rowcount == 1

    def delete_checkpoint_thread(self, checkpoint_thread_id: str) -> None:
        """Delete operational checkpoints after completion or user deletion."""
        self.checkpointer.delete_thread(checkpoint_thread_id)


_store: Optional[PostgresConversationStore] = None
_store_lock = Lock()


def get_memory_store() -> PostgresConversationStore:
    """Lazily initialize the process-wide PostgreSQL pool."""
    global _store  # pylint: disable=global-statement
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = PostgresConversationStore(Config().postgres_uri)
    return _store


def store_conversation(
    thread_id: str, doctor_id: str, user_input: str, response: str
) -> int:
    """Compatibility wrapper used by API, UI and CLI."""
    return get_memory_store().store_turn(
        thread_id, doctor_id, user_input, response
    )


def get_conversation_history(
    thread_id: str, doctor_id: str
) -> List[Dict[str, str]]:
    """Retrieve a doctor's full conversation history from PostgreSQL."""
    return get_memory_store().get_history(thread_id, doctor_id)


def get_all_conversations(doctor_id: str) -> List[str]:
    """Retrieve only the authenticated doctor's thread IDs."""
    return get_memory_store().get_thread_ids(doctor_id)


def delete_conversation(thread_id: str, doctor_id: str) -> bool:
    """Delete a doctor's persisted conversation from PostgreSQL."""
    return get_memory_store().delete_thread(thread_id, doctor_id)
