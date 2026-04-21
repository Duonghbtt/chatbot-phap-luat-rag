# Hệ thống Hỏi Đáp Văn Bản Pháp Luật Đa Tác Tử sử dụng RAG và LangGraph

> Bài tập lớn môn Xử lý Ngôn ngữ Tự nhiên (NLP)  
> Kiến trúc: **Agentic Multi-Agent RAG** với **LangGraph + Qdrant + FastAPI + Streamlit**

## 1. Giới thiệu

Dự án xây dựng một hệ thống chatbot hỏi đáp văn bản pháp luật tiếng Việt có khả năng:

- hiểu câu hỏi pháp lý bằng ngôn ngữ tự nhiên,
- phân loại intent và mức độ rủi ro,
- truy xuất các điều luật liên quan bằng **Hybrid Search**,
- sinh câu trả lời có căn cứ và trích dẫn,
- tự kiểm tra grounding/citation trước khi trả lời,
- chuyển sang **human review** ở các trường hợp nhạy cảm hoặc độ tin cậy thấp.

Khác với pipeline RAG tuyến tính, phiên bản này triển khai theo hướng **LangGraph agentic workflow**: có routing, retry loop, grounding check, checkpoint và interrupt/resume.

---

## 2. Mục tiêu dự án

- Xây dựng pipeline thu thập và chuẩn hóa văn bản pháp luật.
- Xây dựng vector database trên Qdrant và hybrid retrieval cho tiếng Việt.
- Tổ chức hệ thống bằng **stateful graph** trên LangGraph.
- Bổ sung các cơ chế “chuẩn LangGraph hơn”:
  - route theo intent/risk,
  - retry retrieval khi evidence yếu,
  - grounding check trước khi phát hành câu trả lời,
  - human-in-the-loop cho câu hỏi nhạy cảm,
  - resume từ checkpoint theo `thread_id` / `session_id`.

---

## 3. Kiến trúc agentic mức 3

### 3.1 Luồng tổng quát

```text
User Input
   ↓
analyze_node
   ↓
route_node
   ├── clarify_node
   ├── fast_path_node
   └── legal_agent_subgraph
            ↓
     classify_intent_node
            ↓
      rewrite_query_node
            ↓
        retrieve_node
            ↓
         rerank_node
            ↓
   retrieval_check_node
      ├── retry retrieve
      └── generate_draft_node
                 ↓
      grounding_check_node
         ├── retry retrieve
         ├── revise_answer_node
         └── human_review_node
                    ↓
            citation_format_node
                    ↓
             final_answer_node
```

### 3.2 Điểm nổi bật của kiến trúc

- **Stateful:** mọi node cùng đọc/ghi `AgentState`
- **Branching:** route theo intent, độ mơ hồ, risk level
- **Loop:** retrieval có thể chạy lại nếu evidence chưa đủ
- **Quality gates:** kiểm tra grounding và citation trước khi trả lời
- **Human-in-the-loop:** dùng interrupt/resume cho review
- **Persistence:** checkpoint giúp resume sau khi dừng hoặc lỗi

---

## 4. AgentState đề xuất

```python
class AgentState(TypedDict):
    question: str
    normalized_question: str
    intent: str
    intent_confidence: float
    risk_level: str
    need_clarify: bool
    rewritten_queries: list[str]
    retrieved_docs: list[dict]
    reranked_docs: list[dict]
    retrieval_ok: bool
    draft_answer: str
    grounding_ok: bool
    citations: list[dict]
    final_answer: str
    review_required: bool
    history: list
    loop_count: int
    thread_id: str
    session_id: str
```

---

## 5. Công nghệ sử dụng

- **LangGraph**: điều phối workflow đa tác tử dạng đồ thị
- **LangChain**: retriever, prompt, model wrapper
- **Qdrant**: vector database chính, ANN retrieval, payload filtering, alias switching
- **BM25**: keyword retrieval
- **CrossEncoder**: reranking
- **FastAPI**: backend API
- **Streamlit**: giao diện người dùng
- **RAGAS**: đánh giá chất lượng câu trả lời

---

## 6. Cấu trúc thư mục đề xuất

```text
project_root/
├── README.md
├── requirements.txt
├── .env.example
├── configs/
│   ├── app.yaml
│   ├── indexing.yaml
│   ├── retrieval.yaml
│   ├── prompts.yaml
│   └── routing.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── chunks/
│   └── manifests/
│       └── legal_corpus_manifest.jsonl
├── notebooks/
│   ├── 01_eda_legal_corpus.ipynb
│   ├── 02_embedding_benchmark.ipynb
│   ├── 03_prompt_debug.ipynb
│   └── 04_error_analysis.ipynb
├── src/
│   ├── tv1_data/
│   │   ├── crawl_sources.py
│   │   ├── parse_clean.py
│   │   ├── chunk_legal_docs.py
│   │   └── incremental_sync.py
│   ├── tv2_index/
│   │   ├── embedding_registry.py
│   │   ├── build_qdrant_index.py
│   │   ├── qdrant_manager.py
│   │   ├── search_with_filters.py
│   │   └── swap_active_collection.py
│   ├── tv3_retrieval/
│   │   ├── rewrite_query_node.py
│   │   ├── retrieve_node.py
│   │   ├── rerank_node.py
│   │   ├── retrieval_check_node.py
│   │   └── fallback_policy.py
│   ├── tv4_router/
│   │   ├── intent_classifier.py
│   │   ├── route_node.py
│   │   ├── clarify_detector.py
│   │   └── risk_tagger.py
│   ├── tv5_reasoning/
│   │   ├── prompt_library.py
│   │   ├── generate_draft_node.py
│   │   ├── grounding_check_node.py
│   │   ├── revise_answer_node.py
│   │   └── citation_critic.py
│   ├── graph/
│   │   ├── state.py
│   │   ├── builder.py
│   │   ├── subgraphs.py
│   │   ├── human_review_node.py
│   │   └── checkpointing.py
│   └── app/
│       ├── api/
│       │   ├── main.py
│       │   └── routes/
│       │       ├── chat.py
│       │       └── stream.py
│       └── ui/
│           └── streamlit_app.py
├── evaluation/
│   ├── eval_retrieval.py
│   ├── eval_ragas.py
│   └── eval_grounding.py
└── tests/
    ├── test_router.py
    ├── test_retrieval_flow.py
    └── test_graph_resume.py
```

---

## 7. Quy ước triển khai: `.py` hay `.ipynb`?

### Nên dùng `.py` cho phần lõi hệ thống
Toàn bộ phần dưới đây nên viết bằng **Python modules (`.py`)**:

- graph builder,
- các node LangGraph,
- ETL và data pipeline,
- API FastAPI,
- checkpointing,
- evaluation scripts,
- unit tests / integration tests.

**Lý do:** dễ import, tái sử dụng, kiểm thử, Dockerize và triển khai thật.

### Chỉ dùng `.ipynb` cho thí nghiệm
Notebook nên dùng cho:

- EDA dữ liệu,
- benchmark embedding,
- thử prompt,
- phân tích lỗi retrieval/generation,
- trực quan hóa confusion matrix, RAGAS, biểu đồ.

**Không nên** đặt logic vận hành chính của hệ thống LangGraph vào notebook.

---

## 8. Phân công thành viên và file chính

### Phúc (TV1) — Data Engineer
- `crawl_sources.py`: crawl danh sách văn bản, tải HTML và lưu nguồn thô
- `parse_clean.py`: bóc tách nội dung, làm sạch văn bản, chuẩn hóa metadata
- `chunk_legal_docs.py`: chia văn bản theo Điều/Khoản/Điểm để tạo chunk
- `incremental_sync.py`: đồng bộ văn bản mới mà không cần rebuild toàn bộ dữ liệu

### Dũng (TV2) — Embedding & Vector DB
- `embedding_registry.py`: quản lý embedding models và cấu hình encode
- `build_qdrant_index.py`: sinh embedding, build collection article-level/chunk-level và nạp dữ liệu lên Qdrant
- `qdrant_manager.py`: tạo collection, alias, policy cập nhật chỉ mục
- `search_with_filters.py`: semantic search có lọc metadata/ngày/lĩnh vực
- `swap_active_collection.py`: chuyển collection active sang index mới an toàn

### Huy (TV3) — Retrieval Agent
- `rewrite_query_node.py`: rewrite query theo hướng multi-query
- `retrieve_node.py`: hybrid retrieval giữa BM25 và vector search
- `rerank_node.py`: rerank candidate bằng CrossEncoder
- `retrieval_check_node.py`: kiểm tra evidence đã đủ mạnh hay chưa
- `fallback_policy.py`: quyết định retry retrieval / tăng top-k / fallback

### Cường (TV4) — Router + Frontend
- `intent_classifier.py`: phân loại intent và confidence score
- `route_node.py`: quyết định nhánh chạy tiếp theo
- `clarify_detector.py`: phát hiện câu hỏi mơ hồ hoặc thiếu dữ kiện
- `risk_tagger.py`: gắn nhãn risk level cho câu hỏi pháp lý nhạy cảm
- `streamlit_app.py`: giao diện chat, nguồn trích dẫn, routing state, history

### Đại (TV5) — Legal Reasoning Agent
- `prompt_library.py`: prompt theo intent, bằng chứng và loại tác vụ
- `generate_draft_node.py`: sinh bản nháp câu trả lời
- `grounding_check_node.py`: kiểm tra claim có được evidence hỗ trợ hay không
- `revise_answer_node.py`: viết lại khi grounding chưa đạt
- `citation_critic.py`: kiểm tra chất lượng citation và format trích dẫn

### Dương (TV6) — LangGraph Orchestration + FastAPI
- `state.py`: định nghĩa `AgentState`
- `builder.py`: lắp ghép graph, node, edge, routing
- `subgraphs.py`: đóng gói reusable subgraphs
- `human_review_node.py`: triển khai interrupt/resume cho human review
- `checkpointing.py`: lưu checkpoint và resume theo `thread_id/session_id`
- `main.py`, `chat.py`, `stream.py`: backend API đồng bộ và streaming

---

## 9. Cài đặt nhanh

### 9.1 Clone repo

```bash
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_NAME>
```

### 9.2 Tạo môi trường

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 9.3 Cài dependencies

```bash
pip install -r requirements.txt
```

### 9.4 Tạo file môi trường

```bash
cp .env.example .env
```

Điền các biến môi trường cần thiết như:

- `QDRANT_URL`
- `QDRANT_API_KEY` (nếu có)
- `OLLAMA_BASE_URL`
- `OPENAI_API_KEY` (nếu dùng model cloud)
- `LANGCHAIN_TRACING_V2`
- `LANGCHAIN_API_KEY`

---

## 10. Cách chạy hệ thống

### 10.1 Chạy pipeline dữ liệu

```bash
python -m src.tv1_data.crawl_sources
python -m src.tv1_data.parse_clean
python -m src.tv1_data.chunk_legal_docs
```

### 10.2 Build index

```bash
python -m src.tv2_index.build_qdrant_index
```

### 10.3 Chạy API

```bash
uvicorn src.app.api.main:app --reload
```

### 10.4 Chạy Streamlit

```bash
streamlit run src/app/ui/streamlit_app.py
```

---

## 11. Đánh giá

### Retrieval
- Precision@5
- Recall@5
- MRR

### Intent / Routing
- Accuracy
- Macro F1
- F1 theo lớp
- Clarify trigger accuracy
- Human review escalation accuracy

### Generation / Grounding
- Faithfulness
- Answer Relevancy
- Context Recall
- Context Precision
- Grounded answer rate
- Unsupported claim rate
- Citation correctness rate

### System
- E2E latency p50 / p95
- Retry success rate
- Interrupt-resume latency
- Throughput
- Uptime

---

## 12. Test

```bash
pytest tests/
```

Test chính:

- `test_router.py`
- `test_retrieval_flow.py`
- `test_graph_resume.py`

---

## 13. Hướng phát triển tiếp theo

- mở rộng toàn bộ corpus pháp luật Việt Nam,
- tăng cường metadata pháp lý và thời điểm hiệu lực,
- thêm OCR/PDF pipeline,
- nâng cấp legal critic / legal verifier,
- triển khai cloud + autoscaling,
- bổ sung evaluation với chuyên gia pháp lý.

---

## 14. Ghi chú học thuật

Dự án hướng tới việc chứng minh rằng bài toán hỏi đáp pháp luật **phù hợp với LangGraph hơn chain tuyến tính đơn thuần**, vì hệ thống cần:

- state phức tạp,
- routing động,
- retry loop,
- quality gates,
- human-in-the-loop,
- persistence và resume.

---

## 15. Nhóm thực hiện

- **Phúc** — TV1 Data Engineer
- **Dũng** — TV2 Embedding & Vector DB
- **Huy** — TV3 Retrieval Agent
- **Cường** — TV4 Intent Classifier + Frontend
- **Đại** — TV5 Legal Reasoning Agent
- **Dương** — TV6 LangGraph Orchestration + FastAPI

---

## 16. License

Phục vụ mục đích học tập và nghiên cứu nội bộ.
