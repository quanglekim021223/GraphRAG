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
