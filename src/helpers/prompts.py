"""
Common prompts module for Healthcare GraphRAG system.

This module contains shared prompt templates used across different interfaces
(API, UI, CLI) to maintain consistency in system behavior and instructions.
"""

HEALTHCARE_ASSISTANT_PROMPT = """
You are a healthcare assistant. Based on the user's question:
- If the question is about specific patient data, diseases, doctor, hospital, insurance provider, room or treatments, use the 'rag_tool'.
- If the question is general or no specific data is needed, use the 'llm_tool'.
"""


def get_healthcare_system_prompt(conversation_context=None):
    """
    Get the healthcare assistant system prompt with optional conversation context.

    Args:
        conversation_context (str, optional): Previous conversation context to include

    Returns:
        str: Complete system prompt
    """
    prompt = HEALTHCARE_ASSISTANT_PROMPT

    if conversation_context:
        prompt += f"""
        
{conversation_context}

When responding to the user, reference information from previous parts of the conversation when relevant.
You have a complete memory of the conversation history and should maintain continuity.
"""

    return prompt
