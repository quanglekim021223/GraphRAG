# AGENTS.md

## Cursor Cloud specific instructions

### Overview
Healthcare GraphRAG — a Python chatbot combining Neo4j graph database with Azure OpenAI (via GitHub Models endpoint) for medical Q&A. Three interface modes: FastAPI (`:5000`), Streamlit (`:8501`), CLI.

### Services
| Service | How to run | Port |
|---------|-----------|------|
| Neo4j | `sudo docker compose up -d neo4j` (from `/workspace`) | 7474 (browser), 7687 (bolt) |
| FastAPI | `python3 main.py --mode api --port 5000` | 5000 |
| Streamlit | `streamlit run src/routers/ui_router.py --server.headless true` | 8501 |
| CLI | `python3 main.py --mode cli` | — |

### Key caveats
- The `langchain` package (not just `langchain-core`) must be installed for `from langchain.prompts import PromptTemplate` in `src/handlers/llm_manager.py`. It is **not** listed in `requirements.txt` but is required. Install alongside requirements: `pip install -r requirements.txt langchain`.
- Neo4j uses the `neo4j:5.24-enterprise` Docker image with APOC plugin. The container name is `healthcare-neo4j`.
- The `.env` file must exist at the repo root (copy from `.env.example`). Required env vars: `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `GITHUB_TOKEN`, `LANGCHAIN_API_KEY`.
- `NEO4J_URI` should be `bolt://localhost:7687` for local development (the docker-compose exposes port 7687).
- To load data: copy `data/healthcare.csv` into the Neo4j container import directory (`sudo docker cp data/healthcare.csv healthcare-neo4j:/import/healthcare.csv`) then run a LOAD CSV Cypher query.
- No automated test suite exists in this repo. Lint with: `pylint src/ main.py --disable=all --enable=E`.
- `$HOME/.local/bin` must be on PATH for `streamlit`, `uvicorn`, and other pip-installed scripts.
- Chat endpoints require valid `GITHUB_TOKEN` (for Azure OpenAI inference) and `LANGCHAIN_API_KEY` (for LangSmith). Without these, the server starts but chat requests fail.
