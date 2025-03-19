"""
Logging Configuration module for Healthcare GraphRAG system.

This module configures the logging system for the entire application,
setting up consistent log formatting, log levels based on environment variables,
and providing a global logger instance that can be imported by other modules.
"""
import os
import logging


def setup_logging():
    """Configure logging globally."""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    return logging.getLogger(__name__)


logger = setup_logging()
