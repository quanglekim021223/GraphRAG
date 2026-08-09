# Healthcare GraphRAG 🏥 

## Chatbot y tế thông minh sử dụng đồ thị tri thức và AI

![Healthcare GraphRAG](https://img.shields.io/badge/Healthcare-GraphRAG-blue)
![Python](https://img.shields.io/badge/Python-3.9+-green)
![Neo4j](https://img.shields.io/badge/Patient_Data-Neo4j-brightgreen)
![PostgreSQL](https://img.shields.io/badge/Operational_State-PostgreSQL-blue)
![Azure OpenAI](https://img.shields.io/badge/AI-Azure_OpenAI-orange)
![LangChain](https://img.shields.io/badge/Framework-LangChain-yellow)
![LangGraph](https://img.shields.io/badge/Framework-LangGraph-purple)

Healthcare GraphRAG là một hệ thống chatbot thông minh kết hợp cơ sở dữ liệu đồ thị Neo4j với các mô hình ngôn ngữ lớn (LLM) từ Azure OpenAI để trả lời các câu hỏi y tế chính xác và có ngữ cảnh. Dự án sử dụng kỹ thuật Retrieval-Augmented Generation (RAG) dựa trên đồ thị tri thức.

Tài liệu ôn kiến trúc và phỏng vấn: [Hành trình thiết kế Healthcare Agentic GraphRAG](docs/healthcare-agentic-graphrag-review.md).

## 📋 Mục lục

- [Tính năng nổi bật](#-tính-năng-nổi-bật)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Cài đặt và triển khai](#-cài-đặt-và-triển-khai)
- [Giao diện sử dụng](#-giao-diện-sử-dụng)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)

## ✨ Tính năng nổi bật

- **Truy vấn thông minh**: Tự động chuyển đổi câu hỏi ngôn ngữ tự nhiên thành truy vấn Cypher chính xác
- **Cơ chế ReAct Agent**: Bounded orchestration giữa GraphRAG bệnh án, curated guideline corpus và workflow đa nguồn theo policy
- **Đa ngữ**: Hỗ trợ tiếng Việt và tiếng Anh
- **Lưu trữ hội thoại**: PostgreSQL lưu checkpoint, raw turns và rolling summary theo doctor/thread; Neo4j chỉ giữ patient graph
- **Đa nền tảng**: Giao diện web (Streamlit), API (FastAPI) và CLI
- **Hệ thống bộ nhớ**: Duy trì ngữ cảnh và lịch sử hội thoại
- **Giải thích lý luận**: Hiển thị quá trình suy luận thông qua các truy vấn Cypher
## **Kiến trúc hệ thốnge**
![Kiến trúc hệ thống](assets/images/graphrag.png)

## 🏗 Kiến trúc hệ thống

```plaintext
healthcare-graphrag/
├── .env                       # Biến môi trường (Neo4j, PostgreSQL, API keys)
├── .env.example               # Mẫu biến môi trường
├── .gitignore                 # Cấu hình Git ignore
├── schema.cypher              # Define Schema
├── docker-compose.yml         # Neo4j, PostgreSQL, API, UI, CLI
├── docker-entrypoint.sh       # Script khởi động cho containers
├── Dockerfile                 # Cấu hình build image Docker
├── main.py                    # Điểm khởi chạy chính của ứng dụng
├── README.md                  # Tài liệu dự án
├── requirements.txt           # Dependencies Python
│
├── assets/                    # Tài nguyên tĩnh
│   └── images/                # Hình ảnh cho tài liệu và UI
│       ├── 1.png
│       └── graphrag.png
│
├── backup/                    # Thư mục chứa file dump Neo4j
│   └── neo4j.dump             # File dump cơ sở dữ liệu Neo4j
│
├── data/                      # Dữ liệu nguồn
│   └── healthcare.csv         # Dữ liệu y tế dạng CSV
│
├── neo4j/                     # Cấu hình Neo4j
│   └── entrypoint.sh          # Script khởi động cấu hình Neo4j
│
└── src/                       # Mã nguồn chính
    ├── config/                # Cấu hình ứng dụng
    │   ├── settings.py        # Cài đặt cấu hình chính
    │   └── __pycache__/
    │
    ├── handlers/              # Xử lý logic nghiệp vụ
    │   ├── conversation_handler.py   # Quản lý hội thoại
    │   ├── graph_manager.py          # Xử lý đồ thị Neo4j
    │   ├── graphrag_handler.py       # Xử lý GraphRAG
    │   ├── llm_manager.py            # Quản lý LLM
    │   ├── memory_manager.py         # Quản lý bộ nhớ
    │   └── __pycache__/
    │
    ├── helpers/               # Tiện ích hỗ trợ
    │   ├── agent_initializer.py      # Khởi tạo ReAct Agent
    │   ├── llm_initializer.py        # Khởi tạo LLM
    │   └── ...
    │
    └── routers/               # Các giao diện người dùng
        ├── api_router.py       # Giao diện FastAPI
        ├── cli_router.py       # Giao diện dòng lệnh
        └── ui_router.py        # Giao diện Streamlit
```
Healthcare GraphRAG là một ứng dụng theo mô hình kiến trúc phân lớp với các thành phần chính:

1. **Lớp giao diện người dùng**: 
   - Giao diện web tương tác (Streamlit)
   - API RESTful (FastAPI) 
   - Giao diện dòng lệnh (CLI)

2. **Lớp xử lý**:
   - ReAct Agent đưa ra quyết định sử dụng công cụ nào
   - Trình quản lý bộ nhớ và lịch sử hội thoại
   - Cơ chế theo dõi và phân tích việc sử dụng (LangSmith)

3. **Lớp công cụ**:
   - GraphRAG (Truy xuất dữ liệu từ Neo4j và tăng cường câu trả lời)
   - Medical Guideline Tool (Tìm kiếm retrieval-only trên nguồn y khoa allowlist, kèm citation)
   - Patient Guideline Tool (đối chiếu thuốc được nêu rõ với thuốc trong hồ sơ,
     chỉ chuyển medication names đã khử định danh sang curated corpus)

4. **Lớp dữ liệu**:
   - Neo4j Graph Database (nguồn dữ liệu bệnh nhân và quan hệ y tế)
   - PostgreSQL (checkpoint, hội thoại, rolling summary và curated guideline corpus)
   - Azure OpenAI (Mô hình ngôn ngữ)

## 📋 Điều kiện tiên quyết

Để chạy dự án này, bạn cần cài đặt các công cụ sau:
- **Git**: Để clone repository (tải tại [https://git-scm.com/](https://git-scm.com/)).
- **Python 3.9+**: Đảm bảo bạn đã cài Python (tải tại [https://www.python.org/](https://www.python.org/)).
- **Docker**: Cần thiết nếu bạn muốn chạy qua Docker (tải tại [https://www.docker.com/](https://www.docker.com/)).
- **Docker Compose**: Đi kèm với Docker Desktop trên Windows/Mac, hoặc cài riêng trên Linux.

### Cấu hình môi trường
Sao chép file `.env.example` thành `.env` và điền các giá trị:
- `NEO4J_PASSWORD`: Đặt mật khẩu bất kỳ cho Neo4j (ví dụ: `password123`).
- `POSTGRES_URI`: DSN PostgreSQL dùng cho checkpoint, conversation memory và
  curated guideline corpus.
- `LANGCHAIN_API_KEY`: Lấy từ [LangSmith](https://smith.langchain.com/) sau khi đăng ký.
- `GITHUB_TOKEN`: Tạo từ [GitHub Settings](https://github.com/settings/tokens) nếu cần.
- `TAVILY_API_KEY`: Bật tìm kiếm guideline y khoa. Nếu thiếu key, tool sẽ
  fail-closed và không tự tạo câu trả lời không có nguồn.
## Setup

1. Clone the repository
    ```bash
    git clone https://github.com/quanglekim021223/GraphRAG.git
    cd healthcare-graphrag
    ```
2. Create a virtual environment (choose one method):

    Using venv:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

    Using conda:
    ```bash
    conda create -n healthcare-graphrag python=3.9
    conda activate healthcare-graphrag
    ```
3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Download file dump Neo4j
    ```bash
    wget https://mega.nz/file/grA1SaKJ#AzeKD25EmC09aKqKsb0jmGpQYrX3hR6gZqafXqQHjq4 -O backup/neo4j.dump
    ```
## Cấu hình môi trường

Tạo file `.env` với nội dung sau:
```bash
NEO4J_URI=bolt://localhost:7689
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password_here
POSTGRES_URI=postgresql://healthcare:healthcare@localhost:5432/healthcare
MEMORY_RECENT_TURNS=6
MEMORY_SUMMARY_TRIGGER_TURNS=12
MEMORY_SUMMARY_MAX_CHARS=4000
MEMORY_CONTEXT_MAX_CHARS=16000
MEMORY_RAW_RETENTION_DAYS=90

# LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=HealthcareGraphRAG
LANGCHAIN_API_KEY=your_langsmith_api_key_here
# Logging
LOG_LEVEL=INFO
# GitHub token
GITHUB_TOKEN=your_github_token_here
```

## 🐳 Hướng dẫn Docker
Lưu ý quan trọng: Quá trình cần thực hiện 2 bước:

Chạy neo4j-loader để import file dump
Sau đó chạy neo4j và các service khác

### Bước 1: Import dữ liệu với neo4j-loader
Bỏ comment phần neo4j-loader trong file docker-compose.yml
Chạy neo4j-loader để import dữ liệu
```bash
docker-compose up neo4j-loader
```

Đợi cho đến khi thấy thông báo "Database import completed!" và container tự dừng

### Bước 2: Khởi động toàn bộ stack
Khởi động Neo4j và các service khác
```bash
docker-compose up -d
```
Đợi healthcheck của Neo4j và PostgreSQL hoàn tất

### Kiểm tra hoạt động
Kiểm tra các container đang chạy
```bash
docker-compose ps
```
Xem log của Neo4j
```bash
docker-compose logs -f neo4j
```
Kiểm tra dữ liệu trong Neo4j
```bash
docker-compose exec neo4j cypher-shell -u $NEO4J_USERNAME -p $NEO4J_PASSWORD "MATCH (p:Patient) RETURN count(p) AS PatientCount;"
```

### Truy cập các dịch vụ
- **Neo4j Browser**: http://localhost:7474 (đăng nhập với thông tin từ file .env)
- **Streamlit UI**: http://localhost:8501
- **FastAPI**: http://localhost:5000 (hoặc cổng đã cấu hình trong .env)

- **Streamlit**: Giao diện web tương tác để trò chuyện với chatbot.
- **FastAPI**: API RESTful để tích hợp chatbot vào ứng dụng khác.
- **CLI**: Giao diện dòng lệnh để sử dụng nhanh qua terminal.

### Conversation memory

Ba interface dùng chung một đường thực thi và một PostgreSQL store:

```text
request (doctor_id + thread_id)
→ rolling summary + pending/recent turns
→ ReAct với bounded context
→ controlled grounded response
→ persist raw turn
→ completed LangGraph checkpoint cleanup
→ compact older turns khi vượt threshold
```

`conversation_threads` và `conversation_messages` luôn scope bằng khóa kép
`(doctor_id, thread_id)`. Model chỉ nhận summary và một cửa sổ gần đây có hard
character limit; summary chỉ là navigation context, không phải nguồn bằng chứng.
Raw turn chỉ được purge sau khi đã được summary bao phủ và đã quá
`MEMORY_RAW_RETENTION_DAYS`; giá trị này phải theo policy pháp lý/audit của nơi
triển khai. Mọi patient fact trong câu trả lời vẫn phải được truy xuất lại từ
Neo4j trong doctor scope. `PostgresSaver` giữ state bền vững trong lúc graph đang
chạy; toàn bộ checkpoint của thread được xóa sau khi kết quả đã được kiểm soát và
raw turn đã được lưu (hoặc lỗi lưu đã được ghi log), nên nó không trở thành bản
sao lịch sử hội thoại thứ hai.

### Doctor scope authorization

Patient-data queries are fail-closed and require both `patient_id` and
`attending_doctor_id` on every `:Patient` node. The application refuses to start
while any patient is missing either property. Backfill these values from an
authoritative identity/assignment system; never infer ownership with an LLM or
from patient names.

The FastAPI `/chat` endpoint requires an `X-Doctor-ID` header. In production this
header must be overwritten by a trusted authentication gateway or replaced with
the doctor ID extracted from verified JWT/session claims. A client-supplied,
unsigned header is not authentication.

Every generated patient Cypher query is parameterized with `$doctor_id`. Complex
queries such as `UNION`, subqueries and `WITH` pipelines are rejected because the
local scope rewriter intentionally supports only a small auditable subset.
LangGraph checkpoint thread IDs, saved conversations and last-query metadata are
also doctor-scoped/request-local to prevent cross-request history leakage.
The immutable current-turn question is bound to the same request-local security
context, so tools do not trust an LLM-supplied copy that could change a patient
reference before authorization or retrieval.

### Curated medical guideline corpus

General medical questions are answered only from a local corpus of explicitly
reviewed, currently effective documents. The runtime path is:

```text
medical_guideline_tool
→ reject patient-identifying input
→ embed the current de-identified question
→ search approved + active document sections
→ return extractive section text with [G1], [G2] citations
```

Tavily is no longer part of this answer path. It is available only to the admin
`discover` command for finding candidate documents. Provider-side answer
generation and raw-content retrieval remain disabled during discovery. Candidate
URLs are accepted only when HTTPS host and path match the local allowlist in
`src/handlers/medical_guideline_search.py`:

- WHO guidelines, publications and fact sheets
- NICE guidance
- CDC healthcare-professional clinical guidance
- Explicit Ministry of Health document attachment paths

Discovery snippets are never inserted or approved automatically. Patient
identifiers are rejected before any external call, and one chat request may
invoke only one outer data-bearing tool. The bounded composite tool below is the
only exception that can consult both stores internally.

Before a citation is accepted, the backend performs a bounded `HEAD` check with
automatic redirects disabled. Every redirect hop and the final URL must remain
on an approved HTTPS host/path, and DNS answers containing private, loopback,
link-local or reserved addresses are rejected. An infrastructure egress proxy
is still recommended in production to close DNS-rebinding and network-policy
gaps that application code alone cannot fully eliminate.

The discovery runtime also provides process-local controls:

- normalized-question TTL cache;
- per-doctor sliding one-minute rate limit;
- daily provider-call budget;
- one bounded retry for `429`/`5xx`, respecting `Retry-After` only when it is
  below the configured blocking-delay cap;
- circuit breaker after repeated provider failures.

Discovery evidence includes provider URL, validated final URL, retrieval time,
content hash, score and deterministic source priority. These controls are
in-memory per process; multi-worker production deployments must move shared
cache/rate/budget/circuit state to Redis or an equivalent central store.

Curated ingestion is managed with:

```bash
# 1. Find candidates only; nothing is approved automatically.
python3 -m scripts.curated_guidelines discover "WHO hypertension guideline"

# 2. Controlled download and immutable hash; status remains pending_review.
python3 -m scripts.curated_guidelines ingest \
  --url "https://www.who.int/news-room/fact-sheets/detail/hypertension" \
  --title "Hypertension" --publisher "WHO" \
  --publication-date 2025-09-25 --version 2025.1 \
  --effective-from 2025-09-25

# 3. Inspect immutable metadata, document hash and extracted preview sections.
python3 -m scripts.curated_guidelines show DOCUMENT_ID

# 4. Extract, chunk, embed and activate only the exact inspected hash.
python3 -m scripts.curated_guidelines approve DOCUMENT_ID \
  --reviewer REVIEWER_ID --expected-hash SHA256_FROM_SHOW
```

`ingest` downloads at most 10 MB, checks the allowlisted redirect/final URL and
public DNS resolution, accepts only PDF/HTML/plain text, then stores the original
bytes and SHA-256 as `pending_review`. It does not create any vector yet. `show`
extracts preview sections so the reviewer can inspect the frozen content.
`approve` rechecks the exact hash, performs full-text extraction, chunks only
inside section boundaries, creates embeddings, and commits the index plus active
state in one transaction. It records reviewer/time/hash and automatically marks
an older active version of the same source URL as `superseded`. Admins can also
`reject` pending candidates or `withdraw` approved documents.

The catalog, immutable raw bytes and section embeddings are stored in the same
PostgreSQL instance configured by `POSTGRES_URI`, using the independent
`guideline_documents` and `guideline_sections` tables. Schema creation is
idempotent. Embeddings currently use PostgreSQL arrays and deterministic cosine
ranking in Python, so no pgvector extension is required for the bounded corpus.

At query time only documents with `review_status=approved`,
`effective_status=active`, a matching embedding model and a valid effective date
are eligible. Responses contain verbatim section text, title, heading, source
URL, publication date and version; the outer ReAct model cannot rewrite them.
Different sources remain separate and the system does not resolve clinical
disagreement automatically.

### Policy-driven patient-guideline workflow

Questions that require both one authorized patient record and reviewed medical
guidance use one controlled composite path:

```text
patient_guideline_tool
→ select an allowlisted clinical intent
→ GraphRAG lookup for the minimum required facts inside doctor scope
→ policy-filter fields and remove patient identity/history
→ retrieve reviewed, effective guideline sections
→ return patient evidence [E*] and guideline evidence [G*] separately
→ END (return_direct; no outer-agent paraphrase)
```

Supported intents are drug interaction, recorded-condition guidance and
blood-type transfusion compatibility. Medication names must be copied from the
current question; unresolved phrases such as "thuốc này" fail closed to
clarification. Test-result interpretation remains unsupported because the
current graph stores only a generic outcome, without test name, unit or reference
range; associating it with a test named by the user would be ungrounded.
Each intent has a Python allowlist for aliases that may enter the de-identified
handoff. Patient name/ID, doctor, hospital, room, admission, insurance, billing
and conversation history never enter guideline retrieval. Those administrative
facts remain GraphRAG-only. The workflow displays record facts and extractive
guideline sections side by side and does not infer causality, diagnose,
prescribe, or resolve disagreement between sources.

The Python cosine scan is intentionally sized for a small curated corpus. A
large corpus should keep the PostgreSQL metadata contract and move ranking to a
pgvector index or another reviewed vector service.

PostgreSQL integration tests use an isolated temporary schema and require an
explicit test DSN:

```bash
TEST_POSTGRES_URI=postgresql://healthcare:healthcare@localhost:5432/healthcare \
  python3 -m unittest tests.test_curated_guidelines
```
PDFs use outline headings when available and otherwise fall back to page-level
boundaries; scanned PDFs still require a separately reviewed OCR pipeline. An
egress proxy remains necessary to close DNS-rebinding risk completely. Embedding uses
the separately configured GitHub Models endpoint/model; the token needs Models
read permission. Changing the embedding model intentionally makes old vectors
ineligible until the documents are re-ingested with that model.

## Hướng dẫn chạy non-Docker
- **Để chạy Streamlit UI**: 
```bash
python main --mode streamlit
```
- **Để chạy FastAPI API**: 
```bash
python main --mode api
```
- **Để chạy CLI**: 
```bash
python main --mode cli
```

### Giao diện Streamlit
![Streamlit UI Demo](assets/images/1.png)

## 🔄 Khắc phục sự cố

### Lỗi kết nối Neo4j
Nếu gặp lỗi "No node label 'Patient' in the schema":

```bash
# Chạy script cập nhật schema
docker-compose exec neo4j cypher-shell -u $NEO4J_USERNAME -p $NEO4J_PASSWORD "CALL db.schema.visualization();"
docker-compose exec neo4j cypher-shell -u $NEO4J_USERNAME -p $NEO4J_PASSWORD "CALL apoc.meta.schema();"
```

### Lỗi không tìm thấy file dump
Kiểm tra đường dẫn file dump trong thư mục backup:

```bash
ls -la backup/
docker-compose exec neo4j ls -la /backups
```
