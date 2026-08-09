"""
Configuration module for Healthcare GraphRAG system.

This module manages configuration settings loaded from environment variables,
implements a Singleton pattern for configuration access, and validates required
environment variables. It also configures LangChain tracing for observability.
"""
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
    postgres_uri: str = os.getenv(
        "POSTGRES_URI",
        "postgresql://healthcare:healthcare@localhost:5432/healthcare",
    )
    memory_recent_turns: int = int(os.getenv("MEMORY_RECENT_TURNS", "6"))
    memory_summary_trigger_turns: int = int(
        os.getenv("MEMORY_SUMMARY_TRIGGER_TURNS", "12")
    )
    memory_summary_max_chars: int = int(
        os.getenv("MEMORY_SUMMARY_MAX_CHARS", "4000")
    )
    memory_context_max_chars: int = int(
        os.getenv("MEMORY_CONTEXT_MAX_CHARS", "16000")
    )
    memory_raw_retention_days: int = int(
        os.getenv("MEMORY_RAW_RETENTION_DAYS", "90")
    )
    max_result_rows: int = int(os.getenv("MAX_RESULT_ROWS", "20"))
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    medical_search_max_results: int = int(
        os.getenv("MEDICAL_SEARCH_MAX_RESULTS", "3")
    )
    medical_search_min_score: float = float(
        os.getenv("MEDICAL_SEARCH_MIN_SCORE", "0.5")
    )
    medical_search_cache_ttl_seconds: int = int(
        os.getenv("MEDICAL_SEARCH_CACHE_TTL_SECONDS", "300")
    )
    medical_search_rate_limit_per_minute: int = int(
        os.getenv("MEDICAL_SEARCH_RATE_LIMIT_PER_MINUTE", "10")
    )
    medical_search_daily_budget: int = int(
        os.getenv("MEDICAL_SEARCH_DAILY_BUDGET", "1000")
    )
    medical_search_max_retries: int = int(
        os.getenv("MEDICAL_SEARCH_MAX_RETRIES", "1")
    )
    medical_search_max_retry_delay_seconds: float = float(
        os.getenv("MEDICAL_SEARCH_MAX_RETRY_DELAY_SECONDS", "2")
    )
    medical_search_circuit_failure_threshold: int = int(
        os.getenv("MEDICAL_SEARCH_CIRCUIT_FAILURE_THRESHOLD", "3")
    )
    medical_search_circuit_cooldown_seconds: int = int(
        os.getenv("MEDICAL_SEARCH_CIRCUIT_COOLDOWN_SECONDS", "60")
    )
    curated_guideline_db_path: str = os.getenv(
        "CURATED_GUIDELINE_DB_PATH", "data/curated_guidelines.sqlite3"
    )
    curated_embedding_endpoint: str = os.getenv(
        "CURATED_EMBEDDING_ENDPOINT", "https://models.github.ai/inference"
    )
    curated_embedding_model: str = os.getenv(
        "CURATED_EMBEDDING_MODEL", "openai/text-embedding-3-small"
    )
    curated_retrieval_top_k: int = int(
        os.getenv("CURATED_RETRIEVAL_TOP_K", "3")
    )
    curated_retrieval_min_score: float = float(
        os.getenv("CURATED_RETRIEVAL_MIN_SCORE", "0.45")
    )

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize configuration if not already initialized."""
        # Ngăn chặn khởi tạo lại nếu đã tồn tại
        if not hasattr(self, '_initialized'):
            self.github_token = os.getenv("GITHUB_TOKEN", "")
            self.endpoint = "https://models.inference.ai.azure.com"
            self.model_name = "gpt-4o-mini"
            self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7689")
            self.neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
            self.neo4j_password = os.getenv("NEO4J_PASSWORD", "12345678")
            self.postgres_uri = os.getenv(
                "POSTGRES_URI",
                "postgresql://healthcare:healthcare@localhost:5432/healthcare",
            )
            self.memory_recent_turns = int(
                os.getenv("MEMORY_RECENT_TURNS", "6")
            )
            self.memory_summary_trigger_turns = int(
                os.getenv("MEMORY_SUMMARY_TRIGGER_TURNS", "12")
            )
            self.memory_summary_max_chars = int(
                os.getenv("MEMORY_SUMMARY_MAX_CHARS", "4000")
            )
            self.memory_context_max_chars = int(
                os.getenv("MEMORY_CONTEXT_MAX_CHARS", "16000")
            )
            self.memory_raw_retention_days = int(
                os.getenv("MEMORY_RAW_RETENTION_DAYS", "90")
            )
            self.max_result_rows = int(os.getenv("MAX_RESULT_ROWS", "20"))
            self.tavily_api_key = os.getenv("TAVILY_API_KEY", "")
            self.medical_search_max_results = int(
                os.getenv("MEDICAL_SEARCH_MAX_RESULTS", "3")
            )
            self.medical_search_min_score = float(
                os.getenv("MEDICAL_SEARCH_MIN_SCORE", "0.5")
            )
            self.medical_search_cache_ttl_seconds = int(
                os.getenv("MEDICAL_SEARCH_CACHE_TTL_SECONDS", "300")
            )
            self.medical_search_rate_limit_per_minute = int(
                os.getenv("MEDICAL_SEARCH_RATE_LIMIT_PER_MINUTE", "10")
            )
            self.medical_search_daily_budget = int(
                os.getenv("MEDICAL_SEARCH_DAILY_BUDGET", "1000")
            )
            self.medical_search_max_retries = int(
                os.getenv("MEDICAL_SEARCH_MAX_RETRIES", "1")
            )
            self.medical_search_max_retry_delay_seconds = float(
                os.getenv("MEDICAL_SEARCH_MAX_RETRY_DELAY_SECONDS", "2")
            )
            self.medical_search_circuit_failure_threshold = int(
                os.getenv("MEDICAL_SEARCH_CIRCUIT_FAILURE_THRESHOLD", "3")
            )
            self.medical_search_circuit_cooldown_seconds = int(
                os.getenv("MEDICAL_SEARCH_CIRCUIT_COOLDOWN_SECONDS", "60")
            )
            self.curated_guideline_db_path = os.getenv(
                "CURATED_GUIDELINE_DB_PATH", "data/curated_guidelines.sqlite3"
            )
            self.curated_embedding_endpoint = os.getenv(
                "CURATED_EMBEDDING_ENDPOINT",
                "https://models.github.ai/inference",
            )
            self.curated_embedding_model = os.getenv(
                "CURATED_EMBEDDING_MODEL", "openai/text-embedding-3-small"
            )
            self.curated_retrieval_top_k = int(
                os.getenv("CURATED_RETRIEVAL_TOP_K", "3")
            )
            self.curated_retrieval_min_score = float(
                os.getenv("CURATED_RETRIEVAL_MIN_SCORE", "0.45")
            )
            self._initialized = True

    def validate(self) -> None:
        """Validate the configuration."""
        required_vars = ["GITHUB_TOKEN", "NEO4J_URI",
                         "NEO4J_USERNAME", "NEO4J_PASSWORD", "POSTGRES_URI"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(
                f"Missing environment variables: {', '.join(missing_vars)}")

        tracing_enabled = os.getenv(
            "LANGCHAIN_TRACING_V2", "false"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if tracing_enabled and not os.getenv("LANGCHAIN_API_KEY"):
            raise ValueError(
                "LANGCHAIN_API_KEY is required when LangSmith tracing is enabled."
            )

        if self.memory_recent_turns < 1:
            raise ValueError("MEMORY_RECENT_TURNS must be at least 1")
        if self.memory_summary_trigger_turns <= self.memory_recent_turns:
            raise ValueError(
                "MEMORY_SUMMARY_TRIGGER_TURNS must be greater than "
                "MEMORY_RECENT_TURNS"
            )
        if self.memory_summary_max_chars < 256:
            raise ValueError("MEMORY_SUMMARY_MAX_CHARS must be at least 256")
        if self.memory_context_max_chars < self.memory_summary_max_chars:
            raise ValueError(
                "MEMORY_CONTEXT_MAX_CHARS must be at least "
                "MEMORY_SUMMARY_MAX_CHARS"
            )
        if self.memory_raw_retention_days < 1:
            raise ValueError("MEMORY_RAW_RETENTION_DAYS must be at least 1")


# Configure LangSmith only when explicitly enabled. Forcing tracing on can send
# sensitive conversation/tool payloads to an external service unexpectedly.
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv(
    "LANGCHAIN_TRACING_V2", "false"
)
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv(
    "LANGCHAIN_PROJECT", "HealthcareGraphRAG")
