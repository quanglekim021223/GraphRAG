import os
import argparse
from dotenv import load_dotenv
from src.helpers.logging_config import logger  # Cập nhật import

# Load environment variables
load_dotenv()


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='Healthcare GraphRAG System')
    parser.add_argument('--mode', type=str, default='api', choices=['api', 'ui', 'cli'],
                        help='Run mode: api (Flask API), ui (Streamlit UI), or cli (Command Line)')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port for API server')
    args = parser.parse_args()

    try:
        if args.mode == 'api':
            from src.routers.api_router import run_api  # Cập nhật import
            run_api(port=args.port)
        elif args.mode == 'ui':
            from src.routers.ui_router import run_ui  # Cập nhật import
            run_ui()
        elif args.mode == 'cli':
            from src.routers.cli_router import run_cli  # Cập nhật import
            run_cli()
    except ValueError as e:
        logger.error(f"Startup failed: {str(e)}")
        print(f"Error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
