"""
LLM Manager module for Healthcare GraphRAG system.

This module handles interactions with OpenAI language models, including query generation,
validation, and response formatting. It serves as the interface between the application
and external AI services, providing error handling and prompt templating.
"""
from typing import Dict, Any
import re
from openai import OpenAI, OpenAIError
from langchain.prompts import PromptTemplate
from src.helpers.logging_config import logger


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

    def generate_cypher_query(self, question: str, schema: Dict[str, Any]) -> str:
        """
        Generate a Cypher query from a natural language question.

        Args:
            question: User's natural language question
            schema: Neo4j database schema

        Returns:
            Generated Cypher query string

        Raises:
            ValueError: If query generation fails
        """
        prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template="""
            Based on the Neo4j schema:
            {schema}

            Generate an accurate Cypher query to answer: "{question}".
            - Use labels: Patient(name, age, gender, blood_type, admission_type, date_of_admission, discharge_date),
            Disease(name), Doctor(name), Hospital(name), InsuranceProvider(name), Room(room_number),
            Medication(name), TestResults(test_outcome), Billing(amount).
            - Relationships: HAS_DISEASE, TREATED_BY, ADMITTED_TO, COVERED_BY, STAY_IN, TAKE_MEDICATION,
            UNDERGOES, HAS_BILLING, WORKS_AT, PRESCRIBES, RELATED_TO_TEST, PARTNERS_WITH.
            - For name attributes, use case-insensitive matching by applying toLower() on both the node's property and the input value, e.g., WHERE toLower(n.name) = toLower('value').
            - Return only the Cypher query, no markdown or extra text.
            - Ensure valid syntax with MATCH, RETURN, LIMIT 5, matching the schema.
            """
        )
        try:
            response = self.llm.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt.format(
                        schema=schema, question=question)},
                ],
                temperature=0.3,
                max_tokens=1000,
                model=self.config.model_name
            )
            query = response.choices[0].message.content.strip()
            query = re.sub(r"```cypher|```", "", query).strip()
            logger.info("Generated Cypher query: %s", query)
            return query
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

    def generate_response(self, question: str, query_result: Any) -> str:
        """
        Generate a natural language response from query results.

        Args:
            question: Original user question
            query_result: Results from Neo4j database query

        Returns:
            Natural language response

        Raises:
            ValueError: If response generation fails
        """
        prompt = PromptTemplate(
            input_variables=["question", "result"],
            template="""
            Based on the question: "{question}"
            Neo4j Cypher query results: {result}

            Generate a concise, accurate response in English:
            - Use the Cypher query results as the primary source of information.
            - If the query results are empty, state: "No information found from the database."
            - Avoid speculation; stick to the provided data.
            - Format the response appropriately based on the question (e.g., list of patients, details of a disease, etc.).
            """
        )
        try:
            response = self.llm.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt.format(
                        question=question,
                        result=query_result if query_result else "[]"
                    )},
                ],
                temperature=0.3,
                max_tokens=1000,
                model=self.config.model_name
            )
            response_text = response.choices[0].message.content.strip()
            logger.info("Generated response text: %s", response_text)
            return response_text
        except OpenAIError as e:
            logger.error("Failed to generate response: %s", str(e))
            raise ValueError(f"Response generation failed: {str(e)}") from e
