import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Config:
    """Configuration class for Healthcare GraphRAG system."""
    github_token: str = os.getenv("GITHUB_TOKEN", "")
    endpoint: str = "https://models.inference.ai.azure.com"
    model_name: str = "gpt-4o-mini"
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7689")
    neo4j_username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "12345678")

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self, *args, **kwargs):
        # Ngăn chặn khởi tạo lại nếu đã tồn tại
        if not hasattr(self, '_initialized'):
            self.github_token = os.getenv("GITHUB_TOKEN", "")
            self.endpoint = "https://models.inference.ai.azure.com"
            self.model_name = "gpt-4o-mini"
            self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7689")
            self.neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
            self.neo4j_password = os.getenv("NEO4J_PASSWORD", "12345678")
            self._initialized = True

    def validate(self) -> None:
        """Validate the configuration."""
        required_vars = ["GITHUB_TOKEN", "NEO4J_URI",
                         "NEO4J_USERNAME", "NEO4J_PASSWORD"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(
                f"Missing environment variables: {', '.join(missing_vars)}")

        if not os.getenv("LANGCHAIN_API_KEY"):
            raise ValueError(
                "LANGCHAIN_API_KEY must be provided in environment variables.")


# Configure LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv(
    "LANGCHAIN_PROJECT", "HealthcareGraphRAG")
