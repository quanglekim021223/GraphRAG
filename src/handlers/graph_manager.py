from typing import Dict, Any, List
from neo4j.time import Date
from langchain_neo4j import Neo4jGraph
from src.helpers.logging_config import logger
from src.config.settings import Config


class GraphManager:
    def __init__(self, config, *args, **kwargs):
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
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise ValueError(f"Neo4j connection failed: {str(e)}")

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a Cypher query and format the results."""
        try:
            logger.info(f"Executing Cypher query: {query}")
            result = self.graph.query(query)
            records = [
                {k: v.iso_format() if isinstance(v, Date)
                 else v for k, v in record.items()}
                for record in result
            ]
            logger.info(f"Raw query result: {records}")
            return records
        except Exception as e:
            logger.error(
                f"Failed to execute Cypher query: {str(e)}", exc_info=True)
            raise ValueError(f"Query execution failed: {str(e)}")

    def get_schema(self) -> Dict[str, Any]:
        """Get the structured schema from Neo4j."""
        return self.schema
