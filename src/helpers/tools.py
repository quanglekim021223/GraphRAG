"""
Tools module for Healthcare GraphRAG system.

This module provides grounded Neo4j lookup and curated medical-guideline
retrieval. No tool lets the outer ReAct agent rewrite controlled output.
"""
from contextvars import ContextVar
from typing import List, Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from src.config.settings import Config
from src.handlers.curated_guidelines import retrieve_curated_guidelines
from src.helpers.logging_config import logger
from src.handlers.grounding_verifier import (
    format_grounded_response,
)
from src.handlers.graphrag_handler import HealthcareGraphRAG
from src.handlers.patient_guideline_workflow import (
    run_patient_guideline_workflow,
)
from src.helpers.security_context import (
    claim_request_tool,
    get_current_doctor_id,
    get_current_user_question,
)


TOOL_CHAIN_BLOCKED_RESPONSE = (
    "Hệ thống đã chặn việc gọi nhiều công cụ dữ liệu độc lập trong cùng một "
    "lượt. Hãy sử dụng workflow đa nguồn được kiểm soát nếu câu hỏi phù hợp."
)

# Initialize singleton instances
graphrag_instance = HealthcareGraphRAG()

# Request-local Cypher metadata; a process global can leak across doctors.
_LAST_QUERY: ContextVar[Optional[str]] = ContextVar("last_query", default=None)


class PatientGuidelineToolInput(BaseModel):
    """Bounded composite intent selected by the outer tool-calling model."""

    intent: Literal[
        "drug_interaction",
        "disease_guideline",
        "blood_type_compatibility",
    ] = Field(
        description=(
            "Approved clinical purpose for combining patient facts with the "
            "reviewed guideline corpus."
        )
    )
    explicit_terms: List[str] = Field(
        default_factory=list,
        description=(
            "For drug_interaction, medication names explicitly written in the "
            "current question. Empty for all other intents. Never infer terms "
            "from pronouns or conversation history."
        ),
    )


def get_last_query():
    """
    Get the last executed Cypher query.

    Returns:
        str: The most recent Cypher query or None if no query has been executed
    """
    return _LAST_QUERY.get()


def set_last_query(query):
    """
    Set the last executed Cypher query.

    Args:
        query: The Cypher query string to store
    """
    _LAST_QUERY.set(query)


@tool(return_direct=True)
def rag_tool() -> str:
    """Query specific healthcare data using the trusted current question."""
    try:
        if not claim_request_tool("rag_tool"):
            return TOOL_CHAIN_BLOCKED_RESPONSE
        result = graphrag_instance.run(
            get_current_user_question(), get_current_doctor_id()
        )
        logger.info("GraphRAG completed with status: %s", result.get("status"))

        # Lưu query nếu có
        if isinstance(result, dict) and "query" in result:
            set_last_query(result["query"])

        # Return only controlled output; never expose raw Cypher as a fallback.
        if isinstance(result, dict) and "response" in result:
            return format_grounded_response(result)

        return "No information found"
    except Exception as e:  # pylint: disable=broad-exception-caught
        # Broad exception is necessary here as this is a fallback tool
        logger.error("GraphRAG error: %s", str(e))
        return "Không thể truy cập dữ liệu khi chưa xác định danh tính bác sĩ."


@tool(return_direct=True)
def medical_guideline_tool() -> str:
    """Retrieve guidance for the trusted current question from reviewed documents."""
    if not claim_request_tool("medical_guideline_tool"):
        return TOOL_CHAIN_BLOCKED_RESPONSE
    config = Config()
    result = retrieve_curated_guidelines(
        question=get_current_user_question(),
        postgres_uri=config.postgres_uri,
        endpoint=config.curated_embedding_endpoint,
        api_key=config.github_token,
        embedding_model=config.curated_embedding_model,
        top_k=config.curated_retrieval_top_k,
        min_score=config.curated_retrieval_min_score,
    )
    return str(result["response"])


@tool(return_direct=True, args_schema=PatientGuidelineToolInput)
def patient_guideline_tool(
    intent: Literal[
        "drug_interaction",
        "disease_guideline",
        "blood_type_compatibility",
    ],
    explicit_terms: List[str],
) -> str:
    """Combine one authorized patient's allowed facts with reviewed guidance.

    Python policy limits each intent to medication, condition or blood type
    fields. It strips patient identity before guideline retrieval and
    never forwards administrative or financial record fields.
    """
    if not claim_request_tool("patient_guideline_tool"):
        return TOOL_CHAIN_BLOCKED_RESPONSE
    config = Config()
    result = run_patient_guideline_workflow(
        question=get_current_user_question(),
        intent=intent,
        explicit_terms=explicit_terms,
        doctor_id=get_current_doctor_id(),
        graphrag=graphrag_instance,
        guideline_retriever=retrieve_curated_guidelines,
        guideline_options={
            "postgres_uri": config.postgres_uri,
            "endpoint": config.curated_embedding_endpoint,
            "api_key": config.github_token,
            "embedding_model": config.curated_embedding_model,
            "top_k": config.curated_retrieval_top_k,
            "min_score": config.curated_retrieval_min_score,
        },
    )
    logger.info(
        "Patient-guideline workflow completed with status: %s",
        result.get("status"),
    )
    return str(result["response"])
