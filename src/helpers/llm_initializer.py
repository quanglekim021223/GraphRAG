"""
LLM Initializer module for Healthcare GraphRAG system.

This module provides factories for creating and configuring language model instances
used throughout the application. It centralizes LLM configuration to ensure
consistent settings and easy updates.
"""
from langchain_openai import ChatOpenAI
from src.config.settings import Config


def get_llm():
    """
    Create and configure a ChatOpenAI language model instance.

    Uses the application configuration to set up the model with appropriate
    endpoint, API key, model name, and temperature settings.

    Returns:
        ChatOpenAI: Configured language model instance ready for use
    """
    config = Config()
    return ChatOpenAI(
        base_url=config.endpoint,
        api_key=config.github_token,
        model_name=config.model_name,
        temperature=0.3
    )
