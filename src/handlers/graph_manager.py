"""
Graph Manager module for Healthcare GraphRAG system.

This module handles connections to Neo4j graph database, executes Cypher queries,
and provides schema information for the GraphRAG system. It includes error handling
and data formatting for database operations.
"""
from typing import Dict, Any, List
from neo4j.time import Date
from langchain_neo4j import Neo4jGraph
from src.helpers.logging_config import logger


class GraphManager:
    """
    Manages Neo4j graph database connections and operations.

    Provides methods to execute Cypher queries against Neo4j, handle results,
    and retrieve schema information for use in the GraphRAG pipeline.
    """

    def __init__(self, config):
        """
        Initialize GraphManager with configuration.

        Args:
            config: Configuration object containing Neo4j connection details
        """
        self.config = config
        try:
            self.graph = Neo4jGraph(
                url=self.config.neo4j_uri,
                username=self.config.neo4j_username,
                password=self.config.neo4j_password
            )
            self.schema = self.graph.get_structured_schema
            logger.info("Neo4j schema loaded successfully.")
        except Exception as e:
            logger.error("Failed to connect to Neo4j: %s", str(e))
            raise ValueError(f"Neo4j connection failed: {str(e)}") from e

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
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
            logger.info("Executing Cypher query: %s", query)
            result = self.graph.query(query)
            records = [
                {k: v.iso_format() if isinstance(v, Date)
                 else v for k, v in record.items()}
                for record in result
            ]
            logger.info("Raw query result: %s", records)
            return records
        except Exception as e:
            logger.error(
                "Failed to execute Cypher query: %s", str(e), exc_info=True)
            raise ValueError(f"Query execution failed: {str(e)}") from e

    def get_schema(self) -> Dict[str, Any]:
        """
        Get the structured schema from Neo4j.

        Returns:
            Dictionary representation of Neo4j database schema
        """
        return self.schema
