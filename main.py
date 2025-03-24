""" 
Healthcare GraphRAG System - Main entry point.

This module provides the main entry point for the Healthcare GraphRAG system, 
supporting multiple interfaces (API, UI, CLI) with appropriate command line arguments. 
"""
import argparse
import subprocess
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from src.helpers.logging_config import logger
from src.routers.cli_router import run_cli
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
            # Thay đổi: Chạy UI bằng subprocess để gọi trực tiếp streamlit run
            run_streamlit_ui()
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


def run_streamlit_ui():
    """
    Run Streamlit UI by spawning a subprocess.
    This ensures Streamlit runs in its own context, solving the ScriptRunContext issue.
    """
    try:
        # Lấy đường dẫn đến thư mục gốc của dự án
        project_root = Path(__file__).resolve().parent

        # Đường dẫn đến file ui_router.py
        ui_router_path = project_root / "src" / "routers" / "ui_router.py"

        if not ui_router_path.exists():
            raise FileNotFoundError(
                f"UI router script not found at {ui_router_path}")

        # Thiết lập PYTHONPATH để đảm bảo imports hoạt động chính xác
        env = os.environ.copy()
        if 'PYTHONPATH' not in env:
            env['PYTHONPATH'] = str(project_root)
        else:
            env['PYTHONPATH'] = f"{project_root}:{env['PYTHONPATH']}"

        logger.info(
            f"Starting Streamlit UI with PYTHONPATH={env['PYTHONPATH']}")
        logger.info(f"Running: streamlit run {ui_router_path}")

        # Chạy Streamlit trực tiếp thông qua subprocess
        process = subprocess.run(
            ["streamlit", "run", str(ui_router_path)],
            env=env,
            check=True
        )

        return process.returncode
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise ValueError(f"UI startup failed: {str(e)}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Streamlit process failed with exit code {e.returncode}")
        raise ValueError(f"UI startup failed with exit code {e.returncode}")
    except Exception as e:
        logger.error(f"Failed to start Streamlit UI: {str(e)}", exc_info=True)
        raise ValueError(f"UI startup failed: {str(e)}")


if __name__ == "__main__":
    main()
