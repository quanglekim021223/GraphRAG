from typing import Dict, Any, Optional
import re
from openai import OpenAI, OpenAIError
from langchain.prompts import PromptTemplate
from src.helpers.logging_config import logger
from src.config.settings import Config


class LLMManager:
    """Manages interactions with the language model."""

    def __init__(self, config: Config):
        """Initialize the LLM manager with configuration."""
        self.config = config
        try:
            self.llm = OpenAI(base_url=config.endpoint,
                              api_key=config.github_token)
            logger.info("OpenAI client initialized successfully.")
        except OpenAIError as e:
            logger.error(f"OpenAI initialization failed: {str(e)}")
            raise ValueError(f"Failed to initialize OpenAI client: {str(e)}")

    def generate_cypher_query(self, question: str, schema: Dict[str, Any]) -> str:
        """Generate a Cypher query from a natural language question."""
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
            logger.info(f"Generated Cypher query: {query}")
            return query
        except OpenAIError as e:
            logger.error(f"Failed to generate Cypher query: {str(e)}")
            raise ValueError(f"Cypher query generation failed: {str(e)}")

    def validate_cypher_query(self, query: str, schema: Dict[str, Any]) -> str:
        """Validate a Cypher query."""
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
                logger.warning(f"Invalid Cypher query detected: {result}")
                raise ValueError(f"Invalid Cypher query: {result}")
            return query
        except OpenAIError as e:
            logger.error(f"Failed to validate Cypher query: {str(e)}")
            raise ValueError(f"Cypher query validation failed: {str(e)}")

    def generate_response(self, question: str, query_result: Any) -> str:
        """Generate a natural language response from query results."""
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
            logger.info(f"Generated response text: {response_text}")
            return response_text
        except OpenAIError as e:
            logger.error(f"Failed to generate response: {str(e)}")
            raise ValueError(f"Response generation failed: {str(e)}")
