"""
Healthcare GraphRAG System - Main entry point.

This module provides the main entry point for the Healthcare GraphRAG system,
supporting multiple interfaces (API, UI, CLI) with appropriate command line arguments.
"""
import argparse
from dotenv import load_dotenv
from src.helpers.logging_config import logger
from src.routers.cli_router import run_cli
from src.routers.ui_router import run_ui
from src.routers.api_router import run_api

load_dotenv()


def main():
    """
    Parse command-line arguments and start the appropriate interface (API, UI, CLI).

    Supports multiple runtime modes:
    - API mode: Runs a FastAPI server
    - UI mode: Runs a Streamlit web interface
    - CLI mode: Provides a command-line interface

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description='Healthcare GraphRAG System')
    parser.add_argument('--mode', type=str, default='api',
                        choices=['api', 'ui', 'cli'],
                        help='Run mode: api (FastAPI), ui (Streamlit UI), or cli (Command Line)')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port for API server')
    args = parser.parse_args()

    try:
        if args.mode == 'api':
            run_api(port=args.port)
        elif args.mode == 'ui':
            run_ui()
        elif args.mode == 'cli':
            run_cli()
    except ValueError as e:
        logger.error("Startup failed: %s", str(e))
        print(f"Error: {str(e)}")
    except (ImportError, ModuleNotFoundError) as e:
        logger.error("Module import error: %s", str(e))
        print(f"Module error: {str(e)}")
    except Exception as e:  # pylint: disable=broad-exception-caught
        # Vẫn giữ lại broad exception vì đây là điểm vào chính của ứng dụng
        # nhưng thêm disable-comment để pylint không cảnh báo
        logger.error("Unexpected error: %s", str(e), exc_info=True)
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
