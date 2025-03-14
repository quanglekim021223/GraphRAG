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
