"""
Graph Manager module for Healthcare GraphRAG system.

This module handles connections to Neo4j graph database, executes Cypher queries,
and provides schema information for the GraphRAG system. It includes error handling
and data formatting for database operations.
"""
import re
from typing import Dict, Any, List, Optional
from neo4j.time import Date
from langchain_neo4j import Neo4jGraph
from src.helpers.logging_config import logger
from src.handlers.security_guardrails import audit_event


class GraphManager:
    """
    Manages Neo4j graph database connections and operations.

    Provides methods to execute Cypher queries against Neo4j, handle results,
    and retrieve schema information for use in the GraphRAG pipeline.
    """

    _WRITE_CLAUSE_PATTERN = re.compile(
        r"\b(?:CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|CALL|LOAD\s+CSV|"
        r"FOREACH|GRANT|DENY|REVOKE|ALTER|RENAME|START|STOP|TERMINATE)\b",
        re.IGNORECASE,
    )
    _UNKNOWN_SCHEMA_CODES = (
        "UnknownLabelWarning",
        "UnknownRelationshipTypeWarning",
        "UnknownPropertyKeyWarning",
    )

    def __init__(self, config):
        """
        Initialize GraphManager with configuration.

        Args:
            config: Configuration object containing Neo4j connection details
        """
        self.config = config
        try:
            logger.info("Connecting to Neo4j with URI: %s",
                        self.config.neo4j_uri)
            logger.info("Using username: %s", self.config.neo4j_username)
            # Attempt to connect to Neo4j
            self.graph = Neo4jGraph(
                url=self.config.neo4j_uri,
                username=self.config.neo4j_username,
                password=self.config.neo4j_password
            )
            self.graph.query("RETURN 1")  # Test connection
            self.schema = self.graph.get_structured_schema
            self.validate_scope_data_contract()
            logger.info("Neo4j schema loaded successfully.")
        except Exception as e:
            logger.error("Neo4j connection failed: %s", str(e), exc_info=True)
            raise ValueError(
                f"Neo4j connection failed: {str(e)}. Please ensure that the URL, username, and password are correct."
            ) from e

    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and format the results.

        Args:
            query: Cypher query string to execute

        Returns:
            List of dictionaries containing query results

        Raises:
            ValueError: If query execution fails
        """
        try:
            self.validate_read_only(query)
            logger.info("Executing Cypher query: %s", query)
            result = self.graph.query(query, params=parameters or {})
            records = [
                {k: v.iso_format() if isinstance(v, Date)
                 else v for k, v in record.items()}
                for record in result
            ]
            logger.info("Cypher query returned %s record(s)", len(records))
            return records
        except Exception as e:
            logger.error(
                "Failed to execute Cypher query: %s", str(e), exc_info=True)
            raise ValueError(f"Query execution failed: {str(e)}") from e

    def validate_read_only(self, query: str) -> None:
        """Reject empty or potentially mutating Cypher before it reaches Neo4j."""
        if not query or not query.strip():
            raise ValueError("Cypher query is empty")

        query_without_literals = re.sub(
            r"'(?:\\.|''|[^'])*'|\"(?:\\.|\"\"|[^\"])*\"",
            lambda match: " " * len(match.group(0)),
            query,
        )
        if self._WRITE_CLAUSE_PATTERN.search(query_without_literals):
            raise ValueError("Only read-only Cypher queries are allowed")

        if not re.search(r"\bRETURN\b", query, re.IGNORECASE):
            raise ValueError("Read-only Cypher query must contain RETURN")

        self.validate_return_projections(query_without_literals)

    def validate_return_projections(self, query: str) -> None:
        """Allow only direct scalar properties and database-side aggregates."""
        match = re.search(
            r"\bRETURN\b(.*?)(?=\bORDER\s+BY\b|\bSKIP\b|\bLIMIT\b|$)",
            query,
            re.IGNORECASE | re.DOTALL,
        )
        if not match:
            raise ValueError("Cypher query must contain a valid RETURN clause")

        variable_labels = {
            variable: self._camel_to_snake(label)
            for variable, label in re.findall(
                r"\(\s*([A-Za-z_]\w*)\s*:\s*([A-Za-z_]\w*)", query
            )
        }
        for projection in self._split_projections(match.group(1)):
            if self._valid_direct_projection(projection, variable_labels):
                continue
            if self._valid_aggregate_projection(projection, variable_labels):
                continue
            raise ValueError(
                "RETURN items must be direct properties or allowlisted "
                "aggregates with schema-derived aliases"
            )

    @staticmethod
    def _split_projections(return_body: str) -> List[str]:
        """Split comma-separated RETURN items without splitting function args."""
        items = []
        start = 0
        depth = 0
        for index, character in enumerate(return_body):
            if character == "(":
                depth += 1
            elif character == ")":
                depth = max(0, depth - 1)
            elif character == "," and depth == 0:
                items.append(return_body[start:index].strip())
                start = index + 1
        items.append(return_body[start:].strip())
        return [item for item in items if item]

    @classmethod
    def _valid_direct_projection(cls, projection, variable_labels) -> bool:
        match = re.fullmatch(
            r"([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s+AS\s+([A-Za-z_]\w*)",
            projection,
            re.IGNORECASE,
        )
        if not match:
            return False
        variable, prop, alias = match.groups()
        label = variable_labels.get(variable)
        if not label:
            return False
        allowed_aliases = {prop}
        allowed_aliases.add(f"{label}_{prop}")
        return alias.lower() in {item.lower() for item in allowed_aliases}

    @classmethod
    def _valid_aggregate_projection(cls, projection, variable_labels) -> bool:
        count_match = re.fullmatch(
            r"count\(\s*(?:DISTINCT\s+)?([A-Za-z_]\w*|\*)\s*\)"
            r"\s+AS\s+([A-Za-z_]\w*)",
            projection,
            re.IGNORECASE,
        )
        if count_match:
            variable, alias = count_match.groups()
            label = variable_labels.get(variable)
            allowed_aliases = {"count", "result_count"}
            if label:
                allowed_aliases.add(f"{label}_count")
            return alias.lower() in allowed_aliases

        aggregate_match = re.fullmatch(
            r"(sum|avg|min|max)\(\s*(?:DISTINCT\s+)?"
            r"([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s*\)"
            r"\s+AS\s+([A-Za-z_]\w*)",
            projection,
            re.IGNORECASE,
        )
        if not aggregate_match:
            return False
        function, variable, prop, alias = aggregate_match.groups()
        label = variable_labels.get(variable)
        allowed_aliases = {f"{prop}_{function.lower()}"}
        if label:
            allowed_aliases.add(f"{label}_{prop}_{function.lower()}")
        return alias.lower() in allowed_aliases

    @staticmethod
    def _camel_to_snake(value: str) -> str:
        """Convert Neo4j labels such as TestResults to test_results."""
        return re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()

    def explain_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Plan a read-only query and reject unknown schema notifications."""
        self.validate_read_only(query)

        try:
            # pylint: disable=protected-access
            with self.graph._driver.session() as session:
                summary = session.run(
                    f"EXPLAIN {query}", parameters or {}
                ).consume()

            diagnostics = []
            for notification in summary.notifications or []:
                code = notification.get("code", "")
                if any(marker in code for marker in self._UNKNOWN_SCHEMA_CODES):
                    diagnostics.append(
                        notification.get("description")
                        or notification.get("title")
                        or code
                    )

            if diagnostics:
                raise ValueError("; ".join(diagnostics))

            logger.info("Cypher EXPLAIN validation passed: %s", query)
        except ValueError:
            raise
        except Exception as e:
            logger.warning("Cypher EXPLAIN validation failed: %s", str(e))
            raise ValueError(f"Cypher EXPLAIN failed: {str(e)}") from e

    def patient_reference_in_scope(
        self, field_name: str, value: str, doctor_id: str
    ) -> bool:
        """Check an explicit patient reference with parameterized Cypher."""
        if field_name not in {"patient_id", "name"}:
            raise ValueError("Unsupported patient reference field")

        if field_name == "patient_id":
            reference_filter = "toString(p.patient_id) = $patient_reference"
        else:
            reference_filter = (
                "toLower(p.name) = toLower($patient_reference)"
            )
        query = (
            "MATCH (p:Patient) "
            "WHERE p.attending_doctor_id = $doctor_id "
            f"AND {reference_filter} "
            "RETURN count(p) > 0 AS in_scope"
        )
        records = self.graph.query(
            query,
            params={
                "doctor_id": doctor_id,
                "patient_reference": value,
            },
        )
        return bool(records and records[0].get("in_scope"))

    def validate_scope_data_contract(self) -> None:
        """Refuse startup while any Patient lacks mandatory authorization data."""
        query = (
            "MATCH (p:Patient) "
            "WHERE p.patient_id IS NULL "
            "OR trim(toString(p.patient_id)) = '' "
            "OR p.attending_doctor_id IS NULL "
            "OR trim(toString(p.attending_doctor_id)) = '' "
            "RETURN count(p) AS invalid_patient_count"
        )
        records = self.graph.query(query)
        invalid_count = (
            records[0].get("invalid_patient_count", 0) if records else 0
        )
        if invalid_count:
            audit_event(
                "authorization_data_contract_failed",
                level="error",
                invalid_patient_count=invalid_count,
            )
            raise ValueError(
                "Authorization data contract failed: "
                f"{invalid_count} Patient node(s) are missing patient_id or "
                "attending_doctor_id"
            )
        audit_event("authorization_data_contract_passed")

    def get_schema(self) -> Dict[str, Any]:
        """
        Get the structured schema from Neo4j.

        Returns:
            Dictionary representation of Neo4j database schema
        """
        return self.schema
