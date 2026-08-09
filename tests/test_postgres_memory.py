"""Unit tests for bounded PostgreSQL-backed conversation memory."""
import unittest
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.handlers.conversation_handler import CompactionBatch
from src.handlers.memory_manager import (
    ConversationBufferMemory,
    build_conversation_context,
)
from src.helpers.checkpoint_utils import checkpoint_thread_id, current_turn_prompt


class FakeStore:
    """Small in-memory seam; PostgreSQL SQL is covered by runtime smoke tests."""

    def __init__(self, snapshot=None, batch=None):
        self.snapshot = snapshot or {"summary": "", "turns": []}
        self.batch = batch
        self.saved = None
        self.purge_calls = []

    def get_context_snapshot(self, *_args):
        return self.snapshot

    def get_compaction_batch(self, *_args):
        return self.batch

    def save_summary(self, thread_id, doctor_id, batch, summary):
        self.saved = (thread_id, doctor_id, batch, summary)
        return True

    def purge_expired_summarized_turns(
        self, thread_id, doctor_id, retention_days
    ):
        self.purge_calls.append((thread_id, doctor_id, retention_days))
        return 0


class FakeLlm:
    def __init__(self, content="Updated navigation summary"):
        self.content = content
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=self.content)


def memory_config(**overrides):
    values = {
        "memory_recent_turns": 2,
        "memory_summary_trigger_turns": 4,
        "memory_summary_max_chars": 1000,
        "memory_context_max_chars": 4000,
        "memory_raw_retention_days": 90,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class ConversationMemoryTests(unittest.TestCase):
    def test_context_is_explicitly_untrusted_and_not_evidence(self):
        context = build_conversation_context(
            {
                "summary": "User asked a follow-up.",
                "turns": [
                    {
                        "user_input": "What about the previous patient?",
                        "response": "Please clarify the patient ID.",
                    }
                ],
            }
        )

        self.assertIn("untrusted conversation memory", context.lower())
        self.assertIn("not evidence", context.lower())
        self.assertIn("What about the previous patient?", context)

    def test_context_has_a_hard_character_bound(self):
        context = build_conversation_context(
            {
                "summary": "old " * 100,
                "turns": [
                    {"user_input": "x" * 500, "response": "y" * 500}
                ],
            },
            max_chars=350,
        )

        self.assertLessEqual(len(context), 350)
        self.assertIn("older memory truncated", context)

    def test_facade_reads_fresh_snapshot_instead_of_caching_messages(self):
        store = FakeStore(
            snapshot={
                "summary": "Initial summary",
                "turns": [{"user_input": "First", "response": "Answer"}],
            }
        )
        memory = ConversationBufferMemory(
            "thread-1", "doctor-1", store=store, config=memory_config()
        )

        self.assertIn("First", memory.get_chat_history())
        store.snapshot = {
            "summary": "Updated summary",
            "turns": [{"user_input": "Second", "response": "New answer"}],
        }

        context = memory.get_conversation_context()
        self.assertIn("Second", context["conversation"])
        self.assertEqual("Updated summary", context["summary"])

    def test_compaction_advances_only_after_summary_is_generated(self):
        batch = CompactionBatch(
            previous_summary="Previous",
            expected_through_message_id=3,
            new_through_message_id=5,
            turns=[
                {"message_id": 5, "user_input": "Question", "response": "Answer"}
            ],
        )
        store = FakeStore(batch=batch)
        llm = FakeLlm("New summary")
        memory = ConversationBufferMemory(
            "thread-1", "doctor-1", store=store, config=memory_config()
        )

        self.assertTrue(memory.compact(llm))
        self.assertEqual("New summary", store.saved[3])
        self.assertIn("untrusted data", llm.calls[0][0].content.lower())
        self.assertEqual([("thread-1", "doctor-1", 90)], store.purge_calls)

    def test_compaction_is_noop_below_threshold(self):
        store = FakeStore(batch=None)
        llm = FakeLlm()
        memory = ConversationBufferMemory(
            "thread-1", "doctor-1", store=store, config=memory_config()
        )

        self.assertFalse(memory.compact(llm))
        self.assertEqual([], llm.calls)
        self.assertEqual([("thread-1", "doctor-1", 90)], store.purge_calls)


class CheckpointPromptTests(unittest.TestCase):
    def test_only_latest_request_segment_is_sent_to_model(self):
        messages = [
            SystemMessage(content="old system"),
            HumanMessage(content="old question"),
            AIMessage(content="old answer"),
            SystemMessage(content="new bounded context"),
            HumanMessage(content="current question"),
        ]

        selected = current_turn_prompt({"messages": messages})

        self.assertEqual(messages[-2:], selected)

    def test_checkpoint_key_is_scoped_by_doctor_and_thread(self):
        self.assertEqual(
            "doctor-1:thread-1",
            checkpoint_thread_id("thread-1", "doctor-1"),
        )
        self.assertNotEqual(
            checkpoint_thread_id("thread-1", "doctor-1"),
            checkpoint_thread_id("thread-1", "doctor-2"),
        )


if __name__ == "__main__":
    unittest.main()
