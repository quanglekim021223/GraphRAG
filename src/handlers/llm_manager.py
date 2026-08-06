"""
LLM Manager module for Healthcare GraphRAG system.

This module handles interactions with OpenAI language models, including query generation,
validation, and response formatting. It serves as the interface between the application
and external AI services, providing error handling and prompt templating.
"""
import re
from typing import Dict, Any
from openai import OpenAI, OpenAIError
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from src.helpers.logging_config import logger


class CypherGeneration(BaseModel):
    """One-call plan containing the query and its data-free response template."""

    cypher: str = Field(
        description="A read-only Cypher query using explicit RETURN aliases."
    )
    response_template: str = Field(
        description=(
            "Natural-language per-row template. Every dynamic value must be a "
            "Python-style {column_alias} placeholder matching an explicit "
            "RETURN alias in cypher. It must not contain concrete names, "
            "numbers, values, or conditional logic."
        )
    )


class LLMManager:
    """Manages interactions with the language model."""

    def __init__(self, config):
        """
        Initialize the LLM manager with configuration.

        Args:
            config: Configuration object containing API endpoints and keys
        """
        self.config = config
        try:
            self.llm = OpenAI(base_url=config.endpoint,
                              api_key=config.github_token)
            logger.info("OpenAI client initialized successfully.")
        except OpenAIError as e:
            logger.error("OpenAI initialization failed: %s", str(e))
            raise ValueError(
                f"Failed to initialize OpenAI client: {str(e)}") from e

    def generate_cypher_query(
        self, question: str, schema: Dict[str, Any]
    ) -> CypherGeneration:
        """
        Generate a Cypher query from a natural language question.

        Args:
            question: User's natural language question
            schema: Neo4j database schema

        Returns:
            Structured Cypher query and per-row response template

        Raises:
            ValueError: If query generation fails
        """
        prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template="""
            Based on the Neo4j schema:
            {schema}

            Generate an accurate Cypher query and a natural response_template
            to answer: "{question}".
            - Use labels: Patient(name, age, gender, blood_type, admission_type, date_of_admission, discharge_date),
            Disease(name), Doctor(name), Hospital(name), InsuranceProvider(name), Room(room_number),
            Medication(name), TestResults(test_outcome), Billing(amount).
            - Relationships: HAS_DISEASE, TREATED_BY, ADMITTED_TO, COVERED_BY, STAY_IN, TAKE_MEDICATION,
            UNDERGOES, HAS_BILLING, WORKS_AT, PRESCRIBES, RELATED_TO_TEST, PARTNERS_WITH.
            - For name attributes, use case-insensitive matching by applying toLower() on both the node's property and the input value, e.g., WHERE toLower(n.name) = toLower('value').
            - Return scalar properties instead of whole nodes or relationships.
            - Give every returned property a stable semantic alias, e.g.
              RETURN p.name AS patient_name, d.name AS disease_name.
            - Perform counts, sums, averages and date filtering in Cypher, then
              return the computed scalar with a semantic alias.
            - Do not add units. Preserve a unit only when it is already part of
              the stored property name.
            - response_template must use Python-style placeholders such as
              {{patient_name}} and {{patient_age}}. Every placeholder must match
              an explicit alias in the query's RETURN clause exactly.
            - Never put a concrete number, name or data value in the template.
              At this stage the real database values are unknown.
            - The template may contain only natural wording, punctuation,
              ordering and connectors. Do not use if/else or conditional logic.
            - For queries returning multiple rows, write one per-row template;
              the backend will loop over the rows.
            - Ensure valid syntax with MATCH, RETURN, LIMIT 5, matching the schema.
            """
        )
        try:
            response = self.llm.beta.chat.completions.parse(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You generate read-only Neo4j Cypher and a data-free "
                            "natural-language response template."
                        ),
                    },
                    {"role": "user", "content": prompt.format(
                        schema=schema, question=question)},
                ],
                temperature=0.3,
                max_tokens=1000,
                model=self.config.model_name,
                response_format=CypherGeneration,
            )
            generation = response.choices[0].message.parsed
            if generation is None:
                raise ValueError("LLM returned no structured Cypher generation")
            generation.cypher = re.sub(
                r"```(?:cypher)?|```", "", generation.cypher,
                flags=re.IGNORECASE,
            ).strip()
            logger.info("Generated Cypher query: %s", generation.cypher)
            return generation
        except OpenAIError as e:
            logger.error("Failed to generate Cypher query: %s", str(e))
            raise ValueError(
                f"Cypher query generation failed: {str(e)}") from e

    def validate_cypher_query(self, query: str, schema: Dict[str, Any]) -> str:
        """
        Validate a Cypher query.

        Args:
            query: Cypher query to validate
            schema: Neo4j database schema

        Returns:
            The validated query if valid

        Raises:
            ValueError: If query validation fails
        """
        prompt = PromptTemplate(
            input_variables=["schema", "query"],
            template="""
            Based on the Neo4j schema:
            {schema}

            Validate the following Cypher query:
            {query}

            Return a single line:
            - 'VALID' if the query is syntactically and semantically correct.
            - 'INVALID: <brief reason>' if invalid (e.g., 'INVALID: Missing MATCH').
            No additional explanation.
            """
        )
        try:
            response = self.llm.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt.format(
                        schema=schema, query=query)},
                ],
                temperature=0.3,
                max_tokens=100,
                model=self.config.model_name
            )
            result = response.choices[0].message.content.strip()
            if not result.startswith("VALID"):
                logger.warning("Invalid Cypher query detected: %s", result)
                raise ValueError(f"Invalid Cypher query: {result}")
            return query
        except OpenAIError as e:
            logger.error("Failed to validate Cypher query: %s", str(e))
            raise ValueError(
                f"Cypher query validation failed: {str(e)}") from e

    def repair_cypher_query(
        self,
        question: str,
        schema: Dict[str, Any],
        invalid_query: str,
        diagnostic: str,
    ) -> CypherGeneration:
        """Repair a failed read-only Cypher query using its diagnostic."""
        prompt = PromptTemplate(
            input_variables=["question", "schema", "query", "diagnostic"],
            template="""
            Repair the read-only Neo4j Cypher query below.

            Original question:
            {question}

            Live Neo4j schema:
            {schema}

            Invalid query:
            {query}

            Validation or database diagnostic:
            {diagnostic}

            Requirements:
            - Preserve the original question's meaning.
            - Use only labels, relationships and properties from the schema.
            - Produce a read-only query using MATCH/OPTIONAL MATCH, WHERE, WITH,
              RETURN, ORDER BY, SKIP or LIMIT only.
            - Never use CREATE, MERGE, DELETE, DETACH, SET, REMOVE, DROP, CALL,
              LOAD CSV or FOREACH.
            - Include LIMIT 5 unless the query returns one aggregate value.
            - Return scalar properties with stable semantic aliases instead of
              whole nodes or relationships.
            - Compute every requested count, sum, average or date filter in
              Cypher and return the computed scalar with a semantic alias.
            - Do not add units. Preserve a unit only when it is already part of
              the stored property name.
            - Also produce a response_template whose Python-style placeholders
              exactly match explicit aliases in the repaired RETURN clause.
            - Never put concrete numbers, names or data values in the template.
              Do not use conditional logic. The backend applies it once per row.
            """,
        )
        try:
            response = self.llm.beta.chat.completions.parse(
                messages=[
                    {
                        "role": "system",
                        "content": "You repair read-only Neo4j Cypher queries.",
                    },
                    {
                        "role": "user",
                        "content": prompt.format(
                            question=question,
                            schema=schema,
                            query=invalid_query,
                            diagnostic=diagnostic,
                        ),
                    },
                ],
                temperature=0,
                max_tokens=1000,
                model=self.config.model_name,
                response_format=CypherGeneration,
            )
            generation = response.choices[0].message.parsed
            if generation is None:
                raise ValueError("LLM returned no structured Cypher repair")
            generation.cypher = re.sub(
                r"```(?:cypher)?|```", "", generation.cypher,
                flags=re.IGNORECASE,
            ).strip()
            logger.info("Repaired Cypher query: %s", generation.cypher)
            return generation
        except OpenAIError as e:
            logger.error("Failed to repair Cypher query: %s", str(e))
            raise ValueError(f"Cypher repair failed: {str(e)}") from e
