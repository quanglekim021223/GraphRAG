# Ôn tập: Hành trình thiết kế Healthcare Agentic GraphRAG

> Tài liệu này ghi lại trình tự tư duy của dự án: mỗi vòng lặp bắt đầu từ một
> câu hỏi hoặc rủi ro, sau đó mới dẫn tới quyết định kiến trúc. Khi ôn, hãy nhớ
> theo mạch **nguyên nhân → vấn đề → quyết định → trade-off**, không học thuộc
> từng tính năng rời rạc.

## Nguyên tắc khi dùng tài liệu

- Các claim dưới đây được đối chiếu với code hiện tại trong repository.
- Tránh từ tuyệt đối như “hoàn toàn”, “tuyệt đối”, “không thể hallucinate”.
- Phân biệt rõ: code đã implement, unit test đã chạy, live integration đã chạy
  và production đã vận hành là bốn mức bằng chứng khác nhau.
- Không dùng con số accuracy `65% → 90%` khi chưa có evaluation dataset, metric,
  baseline và script tái lập kết quả.

## Phiên bản hiện tại trong một câu

> Đây là một production-minded Healthcare Agentic GraphRAG prototype: ReAct
> chọn một workflow có giới hạn, còn Python enforce authorization, read-only
> Cypher, de-identification, evidence mapping và quyền truy cập guideline.

Ba outer tool hiện tại:

```text
rag_tool
→ tra cứu dữ liệu cụ thể trong Neo4j

medical_guideline_tool
→ truy xuất section từ guideline đã review và còn hiệu lực

patient_guideline_tool
→ workflow đa nguồn có policy cho một số clinical intent đã duyệt
```

ReAct phải chọn đúng **một** outer tool trong mỗi request. Việc cần hai nguồn
không được thực hiện bằng cách để agent tự do gọi hai tool liên tiếp; nó được
đóng gói trong `patient_guideline_tool`.

---

## Vòng 0 — Điểm xuất phát: GraphRAG cơ bản

### Câu hỏi khởi phát

Làm sao cho bác sĩ hỏi bằng ngôn ngữ tự nhiên và lấy được dữ liệu nhiều quan hệ
trong Neo4j?

### Baseline ban đầu

```text
Natural-language question
→ LLM sinh Cypher
→ Neo4j execute
→ LLM diễn đạt kết quả
→ user
```

Graph phù hợp với các câu hỏi quan hệ như bệnh nhân — bệnh — thuốc — bác sĩ —
bệnh viện. Tuy nhiên baseline này mới giải quyết retrieval, chưa giải quyết:

- Cypher sai cú pháp hoặc dùng schema không tồn tại.
- Query ghi dữ liệu hoặc trả cấu trúc khó kiểm chứng.
- Bác sĩ A xem dữ liệu thuộc bác sĩ B.
- LLM tự thêm giá trị hoặc nhận định khi diễn đạt.
- Kiến thức y khoa chung không có nguồn.
- Câu hỏi cần kết hợp bệnh án và guideline.

### Điều cần nói đúng khi phỏng vấn

Repo hiện có Healthcare GraphRAG, nhưng không có một Vector RAG baseline và bộ
evaluation đủ để chứng minh `65% → 90%`. `data/healthcare.csv` là dữ liệu bệnh
án mẫu, không phải QA benchmark.

---

## Vòng 1 — Làm cho LLM-to-Cypher có thể kiểm soát và tự phục hồi

### Câu hỏi khởi phát

Nếu LLM đã nhận live schema mà vẫn sinh Cypher sai thì sao?

Live schema giúp giảm lỗi nhưng không tạo guarantee. LLM vẫn có thể:

- Viết sai cú pháp.
- Dùng label, relationship hoặc property không tồn tại.
- Return whole node thay vì scalar.
- Dùng alias gây hiểu sai dữ liệu.
- Sinh query quá phức tạp để authorization rewriter chứng minh là an toàn.

### Quyết định 1: structured output cho Cypher và template

`LLMManager.generate_cypher_query()` dùng Pydantic `CypherGeneration`:

```python
class CypherGeneration(BaseModel):
    cypher: str
    response_template: str
```

LLM trả query và template trong cùng một call. Đây không phải hai LLM calls.

### Quyết định 2: deterministic read-only validation

`GraphManager.validate_read_only()` chạy bằng Python và kiểm tra:

- Query không rỗng và có `RETURN`.
- Không chứa write/admin clauses như `CREATE`, `MERGE`, `DELETE`, `SET`,
  `DROP`, `CALL`, `LOAD CSV`.
- Projection chỉ là scalar property trực tiếp hoặc aggregate được allowlist.
- Direct property phải có alias suy ra từ schema, ví dụ
  `p.age AS patient_age`.

Lớp này nhanh và chặn query nguy hiểm trước khi phụ thuộc vào Neo4j planner.

### Quyết định 3: authorization scope trước EXPLAIN

`enforce_scope()` inject điều kiện parameterized:

```cypher
p.attending_doctor_id = $doctor_id
```

Query không thể được chứng minh an toàn trong subset được hỗ trợ sẽ bị reject.

### Quyết định 4: Neo4j EXPLAIN trước execute

`GraphManager.explain_query()` gửi:

```cypher
EXPLAIN <scoped query>
```

`EXPLAIN` yêu cầu Neo4j parse và lập execution plan nhưng không chạy query lấy
dữ liệu. Nó giúp phát hiện:

- Lỗi cú pháp/planning.
- Parameter hoặc expression không hợp lệ.
- Notification cho label, relationship hoặc property không tồn tại mà code đã
  chủ động xem là diagnostic.

`EXPLAIN` không thể phát hiện query hợp lệ nhưng trả kết quả rỗng.

### Vì sao không gộp Python validation vào EXPLAIN?

Hai lớp trả lời hai câu hỏi khác nhau:

```text
Python validation
→ Query có thuộc subset read-only và có projection được hệ thống cho phép không?

Neo4j EXPLAIN
→ Neo4j có parse/plan được query này trên schema thật không?
```

Neo4j có thể plan một query mà policy ứng dụng không cho phép. Ngược lại, regex
Python không thể thay thế parser và planner thật của Neo4j.

### Quyết định 5: bounded repair loop

Nếu `EXPLAIN` hoặc execute raise `ValueError`, code gọi
`repair_cypher_query()` với:

```text
original user question
+ live schema
+ previous invalid Cypher
+ Neo4j/validation diagnostic
```

Repair trả cả Cypher mới và template mới. Template cũ được giữ nếu vẫn hợp lệ
với aliases của query mới; nếu không, code thử template mới; cả hai không hợp lệ
thì fallback deterministic render.

Sau repair, query phải đi lại qua `enforce_scope()` và vòng
`EXPLAIN → execute`. `MAX_CYPHER_RETRIES = 2`, nghĩa là tối đa ba attempts:
một lần đầu và hai lần repair.

### Kết quả rỗng được xử lý khác lỗi Cypher

Code hiện tại không repair query rỗng. Nếu execute thành công nhưng không có row,
hệ thống trả `needs_clarification` ngay. Đây là lựa chọn fail-closed: không tự nới
filter hoặc đoán một bệnh nhân khác.

### Flow chính xác hiện tại

```text
LLM sinh Cypher + response_template
→ validate template
→ enforce doctor scope
→ deterministic read-only/projection validation
→ Neo4j EXPLAIN
→ execute
   ├─ ValueError → repair bằng old query + schema + diagnostic
   │              → enforce scope lại → EXPLAIN lại
   ├─ empty rows → clarification
   └─ rows → result validation → grounded rendering
```

### Lưu ý khi trả lời phỏng vấn

Trong `LLMManager` còn method `validate_cypher_query()` dùng LLM, nhưng reachable
GraphRAG flow hiện tại không gọi method này. Không nên nói hệ thống đang dùng một
LLM validation call độc lập.

---

## Vòng 2 — Output control: giảm hallucination ở bước diễn đạt

### Câu hỏi khởi phát

Neo4j trả đúng rows, nhưng làm sao ngăn LLM tự thêm hoặc đảo nghĩa dữ liệu y tế?

### Phương án từng cân nhắc: LLM claims + verifier

```text
Neo4j rows
→ LLM viết answer + claims + evidence references
→ verifier kiểm tra
```

Điểm yếu là evidence reference hợp lệ không chứng minh claim đúng ngữ nghĩa.

Ví dụ:

```json
{
  "claim": "Có 4 kết quả bất thường",
  "evidence": ["row_1", "row_2", "row_3"]
}
```

Ba references đều tồn tại nhưng phép đếm vẫn sai. Hoặc field `test_outcome`
tồn tại nhưng có giá trị `Negative`, trong khi claim viết `Positive`.

Phương án này còn thêm một probabilistic component và thêm latency/cost.

### Quyết định: Python giữ quyền đối với mọi giá trị

`grounding_verifier.py` chỉ render scalar lấy từ rows. Mỗi evidence entry ánh xạ
trực tiếp tới:

```json
{
  "row": 0,
  "field": "patient_age",
  "value": 30
}
```

User thấy `[E1]` cạnh câu; phần Evidence cho biết chính xác cell nào hỗ trợ câu.

### Vấn đề mới: deterministic output quá cứng

```text
Patient name: Alice; Patient age: 30. [E1]
```

Output an toàn nhưng không tự nhiên.

### Giải pháp: response template không chứa data

Trong cùng call sinh Cypher, LLM có thể trả:

```text
Bệnh nhân {patient_name} năm nay {patient_age} tuổi.
```

LLM chưa biết giá trị thật. Python mới gọi `template.format(**row)` sau khi
Neo4j trả data.

### `validate_template()` kiểm tra gì?

- Extract placeholder bằng `\{(\w+)\}`.
- Mọi placeholder phải trùng exact alias trong `RETURN`.
- Static text không được chứa chữ số.
- Không chứa `if/else/nếu/ngược lại`.
- Chỉ cho phép chữ, khoảng trắng và punctuation cơ bản.
- Không được copy string literal từ Cypher vào template.

Nếu template fail, toàn bộ query vẫn có thể chạy; renderer fallback về
`field: value`. Nếu một row thiếu field khi format, chỉ row đó fallback.

### Evidence được gắn như thế nào?

Python đọc placeholders đã dùng trong câu, sau đó tạo một evidence entry cho
chính các row/field/value đó:

```text
- Bệnh nhân Alice năm nay 30 tuổi. [E1]

Evidence:
- [E1] row 0.patient_name="Alice", row 0.patient_age=30
```

Evidence không phải citation do LLM tự nghĩ ra. Nó được backend tạo từ object
row đang render.

### Giới hạn cần nói thẳng

Template validator kiểm tra cấu trúc, không hiểu đầy đủ ngữ nghĩa y khoa của
static prose. Một phrase như “hơi cao” có thể vượt qua character validation dù
row chỉ có số. Prompt yêu cầu wording trung tính, nhưng đây chưa phải semantic
proof. Production nghiêm ngặt hơn nên dùng approved phrase templates theo field
hoặc clinical rule engine cho các nhận định định tính.

Vì vậy nên nói “giảm và kiểm soát bề mặt hallucination”, không nói “loại bỏ hoàn
toàn hallucination”.

---

## Vòng 3 — Result validation: kiểm tra sau khi có rows

### Câu hỏi khởi phát

Query hợp lệ và có quyền truy cập, nhưng output shape hoặc giá trị có thể bất
thường thì sao?

### `validate_result()` hiện kiểm tra

1. Kết quả phải là `list[dict]`.
2. Số row không vượt `MAX_RESULT_ROWS` — mặc định 20.
3. `None` được ghi nhận là missing và renderer không tạo evidence cho giá trị null.
4. Một số range được allowlist để cảnh báo:
   - age: 0–120
   - heart rate: 20–250
   - temperature Celsius: 30–45
5. Nếu user hỏi “mới nhất/gần nhất” nhưng Cypher thiếu `ORDER BY ... DESC` và
   `LIMIT`, hệ thống gắn semantic-mismatch warning.

### Điều lớp này không làm

- Không có rule cho mọi field y khoa.
- Không tự sửa giá trị bất thường.
- Không kết luận một giá trị là chẩn đoán.
- Range warning không chứng minh query đã chọn đúng patient/property.

Đây là sanity check có giới hạn, không phải clinical validation engine.

---

## Vòng 4 — Authorization: scope cứng thay vì tin LLM

### Câu hỏi khởi phát

Nếu bác sĩ hỏi đúng cú pháp nhưng query trả bệnh nhân ngoài phạm vi phụ trách thì
sao?

### Nhận thức kiến trúc

Authorization không nên chỉ là prompt kiểu “hãy nhớ lọc doctor”. Prompt là soft
control; quyền truy cập phải được enforce bằng deterministic code.

### Các lớp hiện tại

#### 1. Startup data contract

`GraphManager.__init__()` gọi `validate_scope_data_contract()`. Application từ
chối khởi tạo nếu bất kỳ `Patient` nào thiếu hoặc để rỗng:

- `patient_id`
- `attending_doctor_id`

#### 2. Early patient-scope lookup

Nếu question có patient ID hoặc tên bệnh nhân rõ ràng,
`patient_reference_in_scope()` chạy parameterized query để kiểm tra bệnh nhân có
thuộc doctor hiện tại không. Đây là early rejection để tiết kiệm cost; nó không
thay thế query-level enforcement.

#### 3. Mandatory query rewriting

`enforce_scope()` yêu cầu explicit named `Patient` node và inject `$doctor_id`.
Nó reject các form không thể chứng minh an toàn như UNION, subquery, boolean
bypass hoặc conflicting doctor scope.

#### 4. Identity nằm ngoài tool arguments

`doctor_id` và current question được bind trong request-local `ContextVar`.
ReAct không được truyền hoặc sửa hai giá trị này qua tool arguments.

#### 5. Conversation/checkpoint isolation

- Neo4j chỉ giữ patient clinical graph, không còn lưu chat history.
- PostgreSQL lưu raw turns và rolling summary bằng khóa kép
  `(doctor_id, thread_id)`.
- LangGraph dùng `PostgresSaver`; checkpoint thread ID được prefix bằng doctor
  ID. Checkpoint tồn tại trong lúc graph chạy để hỗ trợ recovery và được xóa khi
  controlled response đã hoàn tất, tránh lưu trùng conversation history.
- Prompt chỉ nhận summary + pending/recent turns có giới hạn; raw history vẫn giữ
  cho UI/audit nhưng không được nạp toàn bộ vào model.
- Summary là context điều hướng không đáng tin cậy, không phải authorization hay
  medical evidence. Patient fact phải được truy xuất lại từ Neo4j.

#### 6. Một outer data tool mỗi request

`claim_request_tool()` chặn outer agent tự chain nhiều data-bearing tools trong
cùng request. Workflow đa nguồn phải đi qua composite tool được kiểm soát.

### Giới hạn production quan trọng

FastAPI hiện nhận `X-Doctor-ID` header trực tiếp. Code có comment rằng production
phải inject giá trị này từ verified auth claims hoặc API gateway, nhưng repo chưa
có JWT/OIDC identity provider thật. Vì vậy nên nói:

> “Authorization logic và tenant scoping đã được implement, nhưng identity
> proofing ở ingress vẫn là production gap.”

`enforce_scope()` cũng là rewriter dựa trên regex cho một Cypher subset nhỏ, không
phải AST parser tổng quát. Đây là chủ ý fail-closed: giảm flexibility để dễ chứng
minh query được scope, nhưng không nên gọi nó là Cypher security parser hoàn chỉnh.

---

## Vòng 5 — Loại bỏ general LLM answer không grounded

### Câu hỏi khởi phát

Nếu có một `llm_tool` trả lời kiến thức chung từ model memory thì tool đó tạo thêm
giá trị gì?

### Vấn đề

- Không có source citation.
- Không biết publication date hoặc version.
- Thêm LLM call nhưng không thêm retrieval capability đáng tin.
- Làm yếu nguyên tắc “không dùng general knowledge làm fallback cho patient data”.

### Quyết định

General medical questions không còn được trả lời bằng model memory. Chúng phải đi
qua nguồn y khoa đã kiểm soát hoặc abstain.

---

## Vòng 6 — Phase 1: live medical search có giới hạn

### Câu hỏi khởi phát

Làm sao thêm kiến thức y khoa chung có nguồn mà không biến search engine thành
nguồn tin mặc định?

### Capability được xây dựng

`medical_guideline_search.py` có live Tavily discovery với:

- Exact host + path allowlist cho một số nguồn WHO, NICE, CDC và Bộ Y tế.
- HTTPS, port và URL-shape checks.
- Tắt automatic redirect; kiểm tra từng hop và final URL.
- DNS public-address check để giảm SSRF risk.
- PII/sensitive-data gate trước network call.
- `include_answer=False`, `include_raw_content=False`.
- Bounded retry, cache, per-actor rate limit, daily budget, circuit breaker.
- Deterministic source priority; không tự hòa giải nguồn mâu thuẫn.

### Trade-off phát hiện

Snippet ngắn có thể mất điều kiện hoặc ngoại lệ. Search result cũng không luôn có
publication date/version đáng tin. Vì vậy live search phù hợp để discovery, chưa
phù hợp làm answer path chính trong healthcare.

---

## Vòng 7 — Phase 2: trusted full-document ingestion

### Câu hỏi khởi phát

Làm sao tự động hóa coverage từ nguồn chính thống mà không dùng snippet Tavily
làm evidence và không bắt con người duyệt mọi URL?

### Pipeline hiện tại

```text
local approved/trusted retrieval
→ nếu miss: Tavily tìm tối đa 5 candidate trong exact whitelist
→ validate URL/redirect/publication date + rank theo source priority và score
→ controlled full-document download theo thứ tự xếp hạng
→ auto-ingest tối đa 3 documents hợp lệ
→ SHA-256 snapshot version + deduplication
→ full-text extraction
→ parse HTML/PDF/plain-text hierarchy thành parent sections
→ sentence-aware child chunks theo budget 400 tokens, overlap 50 tokens
→ prepend document title + section path rồi embedding child
→ PostgreSQL document + parent/child/position metadata
→ auto-activate với review_status=trusted_official
→ retrieve top-k 3 child chunks
→ expand bounded parent hoặc previous/next child + [G*]
```

### Invariants quan trọng

- Tavily snippet không bao giờ được lưu làm corpus content.
- Auto-ingestion chỉ nhận exact host/path allowlist, final redirect hợp lệ,
  publication date không nằm trong tương lai và content type được hỗ trợ.
- Full document được download lại bằng backend, hash và version theo content.
- Version mới của cùng source URL sẽ supersede bản active cũ.
- Retrieval chỉ đọc `approved` hoặc `trusted_official` đang `active`, đúng
  embedding model và nằm trong effective date window.
- Manual `pending_review → approved` vẫn tồn tại nếu policy nội bộ yêu cầu
  duyệt một tài liệu chính thức trước khi sử dụng.
- Output là extractive section text kèm title, heading, URL, publication date,
  version và `[G*]`; ReAct không paraphrase lại.

### Vai trò Tavily hiện tại

Tavily nằm trong conditional runtime fallback khi local corpus miss và vẫn có
admin command `discover`. Nó chỉ tìm URL; backend download full document rồi
index vào PostgreSQL trước khi retry. Không có fallback ra ngoài whitelist.

### Giảm cold-miss bằng warm corpus

CLI `python3 -m scripts.curated_guidelines prewarm` chạy ngoài request path với
taxonomy mặc định gồm 10 nhóm y khoa phổ biến. Mỗi topic đi qua chính pipeline
corpus-first hiện có:

```text
topic de-identified
→ corpus đã cover: không gọi Tavily
→ corpus miss: discovery tối đa 5 URL
→ validate/rank và ingest tối đa 3 documents
→ persist sections + embeddings vào PostgreSQL
```

Có thể truyền nhiều `--topic` để prewarm theo nhu cầu thực tế. Command dừng sớm
khi Tavily unavailable, rate-limited, hết daily budget hoặc circuit mở. Repo
không tự khởi chạy scheduler; production có thể gọi command bằng cron hoặc
deployment scheduler. Long-tail query chưa được warm vẫn đi qua synchronous
cold-miss fallback, nên request đầu tiên của chủ đề đó có thể chậm.

### Giới hạn

- Publication date của auto-ingestion phụ thuộc metadata Tavily; thiếu hoặc nằm
  trong tương lai thì candidate bị loại. Hash chứng minh integrity/snapshot,
  không chứng minh nội dung đúng về mặt lâm sàng hay còn là khuyến nghị mới nhất.
- Corpus miss đầu tiên có latency của search + full download + embedding; các
  request sau dùng document đã persist.
- PostgreSQL lưu raw document, review/version metadata, parent hierarchy và
  child embeddings; cosine hiện vẫn được tính bằng Python nên phù hợp corpus nhỏ.
- Corpus lớn cần pgvector hoặc một reviewed vector service, không cần đổi
  metadata/review contract.
- Scanned PDF cần OCR pipeline riêng.
- Process-local cache/rate-limit/circuit state không phải shared distributed state.

---

## Vòng 8 — Từ router sang bounded orchestrator

### Câu hỏi khởi phát

Nếu bác sĩ hỏi “Ibuprofen có tương tác với thuốc bệnh nhân Alice đang dùng
không?”, một tool patient-only hoặc guideline-only đều không đủ.

### Thiết kế nguy hiểm nếu cho ReAct tự do chain

```text
ReAct gọi rag_tool
→ nhìn thấy patient record
→ tự viết query gửi sang guideline tool
→ tự tổng hợp kết luận
```

Agent có thể đưa tên/ID/room/history ra ngoài, chọn sai field hoặc tự tạo clinical
conclusion.

### Quyết định: một composite workflow có policy

```text
ReAct chọn patient_guideline_tool + typed intent
→ Python xác thực patient reference/explicit terms
→ GraphRAG truy xuất minimum patient facts trong doctor scope
→ policy chỉ lấy aliases được phép
→ privacy check + de-identification
→ curated guideline retrieval
→ hiển thị patient evidence [E*] và guideline evidence [G*] riêng
→ END
```

Các intent hiện được implement:

| Intent | Patient fields được handoff | Điều kiện |
|---|---|---|
| `drug_interaction` | medication aliases | Tên thuốc phải xuất hiện exact trong current question |
| `disease_guideline` | disease/condition aliases | Không nhận explicit term do agent tự thêm |
| `blood_type_compatibility` | `blood_type` | Chỉ dùng cho transfusion compatibility guidance |

Không chuyển sang guideline retrieval:

- Patient name/ID và doctor identity.
- Hospital, room, admission/discharge.
- Insurance và billing.
- Conversation history.

Các câu hỏi hành chính vẫn chỉ đi qua `rag_tool`.

### Vì sao chưa hỗ trợ test-result interpretation?

Dataset hiện chỉ có `TestResults(test_outcome)`/cột `Test Results`, không có:

- test name
- unit
- reference range
- timestamp riêng của test

Nếu user nói “INR” mà graph chỉ trả `Abnormal`, hệ thống không chứng minh được hai
thông tin thuộc cùng xét nghiệm. Intent đó bị reject thay vì tạo association
không grounded.

### ReAct hiện là gì?

Không còn chỉ là keyword router, vì nó phải:

- Phân biệt single-source và multi-source request.
- Chọn một clinical intent có kiểu dữ liệu.
- Copy explicit medication terms từ current question.
- Chọn clarification khi thiếu dữ kiện.

Nhưng nó không phải autonomous orchestrator vì không được tự quyết định tool
chain hoặc data handoff. Cách mô tả chính xác:

> **Policy-aware bounded orchestrator**: ReAct chọn workflow/intent; deterministic
> Python orchestration thực thi và enforce security boundary.

---

## Kiến trúc runtime hiện tại

### 1. Patient factual query

```text
API/UI/CLI
→ bind doctor_id + current question vào ContextVar
→ ReAct chọn rag_tool
→ input guardrail + early patient scope lookup
→ LLM sinh Cypher + template từ live schema
→ template validation
→ doctor-scope enforcement
→ read-only validation + EXPLAIN
→ execute
→ result validation
→ Python grounded renderer + [E*]
→ return_direct
→ controlled response selector
→ user
```

### 2. General medical guideline query

```text
ReAct chọn medical_guideline_tool
→ current question lấy từ trusted ContextVar
→ sensitive-data gate
→ embed query
→ retrieve approved/trusted-official + active + effective sections
→ miss thì Tavily lấy 5 URL trong whitelist, validate + rank
→ auto-ingest tối đa 3 full documents
→ retry local retrieval top-k 3 sections
→ extractive response + [G*]
→ return_direct
→ user
```

### 3. Patient + guideline query

```text
ReAct chọn patient_guideline_tool(intent, explicit_terms)
→ policy-driven GraphRAG lookup
→ alias allowlist
→ de-identification/privacy gate
→ curated section retrieval
→ side-by-side [E*] + [G*]
→ return_direct
→ user
```

---

## Input guardrail và output guardrail nằm ở đâu?

### Input side

- `check_prompt_injection()` tại entrypoint và GraphRAG path.
- `doctor_security_context()` bind identity/current question.
- `check_input_scope()` kiểm tra explicit patient reference.
- `enforce_scope()` là authorization enforcement ở query level.
- `contains_sensitive_patient_data()` chặn patient data đi vào external/general
  guideline path.

### Output side

- `validate_result()` kiểm tra shape, row limit, missing và một số warning.
- `validate_template()` kiểm tra template structure.
- `build_grounded_output()` render từ raw row values và tạo `[E*]`.
- `retrieve_curated_guidelines()` trả extractive sections và `[G*]`.
- `select_controlled_agent_response()` lấy tool output cuối cùng và reject direct
  answer do ReAct tự viết.

Authorization không nên gọi đơn thuần là “input guardrail”; đó là một hard policy
enforcement riêng.

---

## Các script trả lời phỏng vấn ngắn

### “Flow LLM-to-Cypher của em thế nào?”

> “LLM nhận live schema và trả structured output gồm Cypher cùng data-free
> response template trong một call. Python kiểm tra template, inject doctor scope
> và giới hạn query vào read-only scalar projection. Sau đó Neo4j EXPLAIN query
> trước khi execute. Nếu validation hoặc database trả diagnostic, hệ thống đưa
> original question, live schema, query cũ và diagnostic vào repair call, tối đa
> hai retries. Rows rỗng không được tự broaden mà trả clarification.”

### “EXPLAIN khác deterministic validation thế nào?”

> “Python validation enforce application policy như read-only clauses và allowed
> RETURN shape. EXPLAIN dùng parser/planner thật của Neo4j để kiểm tra query có
> plan được trên schema hiện tại không. Một query có thể hợp lệ với Neo4j nhưng
> vẫn bị policy ứng dụng từ chối, nên em giữ hai lớp.”

### “Làm sao giảm hallucination?”

> “Em không cho LLM viết database values ở output stage. LLM chỉ tạo data-free
> sentence template; Python inject raw row values, tạo evidence map về exact
> row/field/value và fallback deterministic nếu template fail. Với guideline, em
> trả extractive approved sections. Tuy nhiên template validation chưa phải
> semantic verifier, nên em mô tả đây là giảm hallucination surface chứ không
> tuyên bố loại bỏ tuyệt đối.”

### “ReAct là router hay orchestrator?”

> “Ban đầu nó gần như router. Hiện tại em gọi nó là bounded orchestrator: agent
> phân biệt single-source/multi-source, chọn workflow và typed intent. Nhưng agent
> không được tự chain data tools. Python workflow mới điều phối GraphRAG,
> de-identification và curated retrieval theo policy allowlist.”

### “Vì sao không tạo một tool cho mỗi field?”

> “Một tool cho thuốc, bệnh, nhóm máu, bệnh viện và hóa đơn sẽ làm tool surface
> phình to, duplicate authorization/privacy logic và khiến routing khó audit. Em
> dùng một composite tool với intent allowlist. Chỉ field lâm sàng cần guideline
> mới được handoff; administrative fields vẫn GraphRAG-only.”

### “Khi guideline và patient data kết hợp, hệ thống có kết luận điều trị không?”

> “Không. Workflow hiển thị patient facts và guideline sections thành hai evidence
> namespaces riêng. Nó không tự suy ra chẩn đoán, quan hệ nhân quả hay thay đổi
> điều trị. Đây là retrieval support, không phải clinical decision engine.”

### “Authorization đã production-ready chưa?”

> “Query-level scope, startup data contract và conversation isolation đã có.
> Nhưng ingress hiện nhận X-Doctor-ID, nên production cần JWT/OIDC hoặc gateway
> inject verified claims. Cypher scope rewriter cũng cố ý chỉ hỗ trợ một subset
> nhỏ; nếu cần Cypher rộng hơn em sẽ chuyển sang AST-based validation.”

---

## Bảng bằng chứng: nói được gì và chưa nên nói gì

| Claim | Trạng thái hiện tại |
|---|---|
| LLM sinh Cypher + template trong một structured call | Có trong code |
| Read-only validation, EXPLAIN, repair tối đa 2 retries | Có trong code và unit tests |
| Doctor-scoped parameterized Cypher | Có trong code và unit tests |
| Startup fail nếu Patient thiếu authorization fields | Có trong code và unit tests |
| Evidence `[E*]` map exact row/field/value | Có trong code và unit tests |
| Approved/trusted-official section retrieval `[G*]` | Có trong code và PostgreSQL integration tests |
| Tavily conditional auto-ingestion trên corpus miss | Có trong reachable code path và integration test |
| Policy-driven patient + guideline workflow | Có cho 3 intents |
| PostgreSQL doctor-scoped memory + rolling summary | Có trong code, unit test và local integration smoke |
| Test-result clinical interpretation | Chưa hỗ trợ do data contract thiếu |
| Identity provider/JWT thật | Chưa có |
| Distributed cache/rate limit | Chưa có |
| Live Neo4j + LLM end-to-end đã verify trong lần kiểm gần nhất | Chưa verify |
| Accuracy tăng 65% → 90% | Chưa có evaluation evidence |
| Production clinical validation | Chưa có |

Snapshot kiểm thử tại thời điểm cập nhật tài liệu: 107 unit tests pass với fake
GraphRAG/retriever và các isolated components. Đây không thay thế live integration
test hoặc clinical evaluation.

---

## Muốn claim accuracy thì cần làm gì?

Tạo một evaluation set versioned, ví dụ:

```text
normal single-hop questions
medium relationship questions
multi-hop questions
authorization/adversarial questions
empty/ambiguous questions
```

Mỗi sample cần expected answer hoặc expected graph facts. So sánh Vector RAG và
GraphRAG trên cùng dataset, cùng policy và cùng judging method.

Metric nên tách riêng:

- Retrieval/answer exactness hoặc token F1 cho factual questions.
- Execution accuracy của Cypher.
- Grounded claim precision.
- Authorization violation rate.
- Clarification/abstention correctness.
- Latency p50/p95 và token/API cost.

Chỉ sau đó mới nói con số phần trăm và phải nêu test size, split, metric, cách
judge và confidence/cherry-picking control.

---

## Định vị dự án

### Cách gọi phù hợp

> A policy-controlled, evidence-grounded Healthcare Agentic GraphRAG prototype
> with scoped Neo4j retrieval and reviewed clinical-guideline workflows.

### Đánh giá thận trọng

| Khía cạnh | Mức hợp lý để trình bày |
|---|---|
| Implementation breadth | Mid-level AI Engineer project |
| Safety/architecture reasoning | Có yếu tố senior-leaning |
| Production readiness | Production-minded prototype |
| Clinical product | Chưa đủ bằng chứng để gọi production clinical system |

### Câu mở đầu “Kể về dự án”

> “Ban đầu đây là một GraphRAG prototype chuyển câu hỏi tự nhiên thành Cypher.
> Em phát triển nó theo các failure mode thực tế: query sai, data leakage giữa
> bác sĩ, hallucination ở output, nguồn guideline thiếu version và câu hỏi cần
> kết hợp nhiều nguồn. Phiên bản hiện tại dùng ReAct như bounded orchestrator,
> còn authorization, evidence mapping, de-identification và curated retrieval
> được enforce bằng deterministic Python. Em gọi nó là production-minded
> prototype vì vẫn thiếu identity provider thật, benchmark trên evaluation set
> và live clinical validation.”

---

## Sợi chỉ đỏ cần nhớ

1. **Fail closed:** không chắc thì reject, abstain hoặc clarification.
2. **Evidence first:** database claim phải map về row/field/value; guideline claim
   phải trỏ về approved section/version.
3. **LLM đề xuất, code quyết định:** LLM tạo query/template/intent; Python enforce
   quyền, data boundary và output contract.
4. **Minimum necessary data:** composite workflow chỉ lấy field cần cho intent.
5. **Discovery khác trust:** search engine tìm candidate; reviewer quyết định
   document nào được index.
6. **Agent autonomy có giới hạn:** càng gần patient data, càng ít tự do tool chain.
7. **Nói đúng bằng chứng:** unit tests không đồng nghĩa production; prototype
   safeguard không đồng nghĩa clinical validation.

---

## Checklist tự luyện

- [ ] Giải thích được deterministic validation và EXPLAIN không trùng nhau.
- [ ] Giải thích được ba attempts khi `MAX_CYPHER_RETRIES = 2`.
- [ ] Nói đúng rằng empty result hiện clarification ngay, không tự broaden.
- [ ] Giải thích evidence `[E*]` được Python tạo từ row/field/value.
- [ ] Nêu được giới hạn semantic của response template.
- [ ] Phân biệt early scope lookup với mandatory query scope enforcement.
- [ ] Chủ động nêu gap `X-Doctor-ID` chưa phải verified identity.
- [ ] Giải thích Tavily chỉ tìm URL, còn backend download full document để index.
- [ ] Giải thích `trusted_official` khác internal `approved` và vì sao snippet
  không được dùng làm corpus content.
- [ ] Nêu ba composite intents hiện được hỗ trợ.
- [ ] Giải thích vì sao test-result interpretation chưa được hỗ trợ.
- [ ] Mô tả ReAct là policy-aware bounded orchestrator.
- [ ] Không dùng claim `65% → 90%` trước khi có eval reproducible.
- [ ] Tách rõ code implemented, tests passed, live verified và production proven.

## Code map để ôn theo repository

```text
src/handlers/llm_manager.py
→ structured Cypher generation and repair

src/handlers/graph_manager.py
→ Neo4j connection, read-only projection validation, EXPLAIN, execute,
  scope data contract

src/handlers/security_guardrails.py
→ input heuristics, doctor-scope enforcement, result validation

src/handlers/grounding_verifier.py
→ template validation, deterministic rendering, evidence, controlled selection

src/helpers/security_context.py
→ trusted request-local doctor/question and one-tool claim

src/helpers/tools.py + src/helpers/prompts.py
→ three outer tools and bounded routing contract

src/handlers/medical_guideline_search.py
→ allowlisted live discovery and operational controls

src/handlers/curated_guidelines.py + scripts/curated_guidelines.py
→ reviewed/versioned ingestion and section retrieval

src/handlers/patient_guideline_workflow.py
→ policy-driven multi-source orchestration and de-identification
```
