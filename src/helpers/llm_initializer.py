from src.config.settings import Config
from langchain_openai import ChatOpenAI


def get_llm():
    config = Config()
    return ChatOpenAI(
        base_url=config.endpoint,
        api_key=config.github_token,
        model_name=config.model_name,
        temperature=0.3
    )
