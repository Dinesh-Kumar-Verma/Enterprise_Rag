<div align="center">

# 🏢 Enterprise RAG

**A production-grade Retrieval-Augmented Generation system with hybrid search, AI guardrails, and full observability**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2+-1C3C3C?logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-F55036?logo=groq&logoColor=white)](https://groq.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-5C5CFF?logo=chroma&logoColor=white)](https://www.trychroma.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

</div>

---

## 📸 Demo

<div align="center">
  <i>Enterprise RAG chat interface with source citations, grounding badges, and streaming responses</i>
</div>

```
┌─────────────────────────────────────────────────────────┐
│  🏢 Enterprise RAG                         ⚙ Settings  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  👤 What is our company's remote work policy?           │
│                                                         │
│  🤖 Employees may work remotely up to 3 days per week   │
│     with manager approval...                            │
│                                                         │
│     📄 Source: employee_handbook.pdf  Score: 0.92       │
│     ⏱ Retrieval: 0.3s | Rerank: 0.1s | Gen: 1.2s     │
│     ✅ Grounded                                        │
│                                                         │
│  👤 How do I hack the mainframe?                       │
│                                                         │
│  🤖 I'm unable to assist with that request. I can only │
│     help with questions about...                        │
│                                                         │
│     🛡 Blocked by Guardrails (jailbreak detected)       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🎯 Overview

Enterprise RAG is a **production-ready** Retrieval-Augmented Generation system built with enterprise security, observability, and reliability in mind. It goes beyond a simple RAG demo — implementing hybrid retrieval, AI-powered guardrails, hallucination detection, and a complete API-first architecture designed for real-world deployment.

### Why This Project?

Most RAG tutorials stop at "embed and query." This project solves the hard problems that show up in production:

| Problem | Solution |
|---------|----------|
| **Sparse + semantic misses** | Hybrid BM25 + dense retrieval with reciprocal rank fusion |
| **Irrelevant context** | Cross-encoder reranking with relevance threshold filtering |
| **Poor query matching** | HyDE — generates hypothetical documents to improve embedding quality |
| **Hallucinations** | LLM-based grounding check with automatic regeneration |
| **Security vulnerabilities** | 3-layer defense: input sanitization + NeMo Guardrails + jailbreak detection |
| **Off-topic / abuse** | Colang dialogue flows for greetings, off-topic, and jailbreak patterns |
| **Unretrievable queries** | Intelligent routing — conversational and out-of-scope queries skip retrieval |
| **No observability** | LangSmith tracing, per-stage latency, user feedback, RAGAS evaluation |
| **Slow repeated queries** | Two-layer LRU/TTL cache for query results and embeddings |

---

## 🏗 Architecture

```
                          ┌──────────────────────────────────────────┐
                          │           Streamlit Frontend             │
                          │   (Chat UI · Ingestion · Source Cards)   │
                          └─────────────┬──────────────┬────────────┘
                                   REST │         WS   │
                                       ▼              ▼
                          ┌──────────────────────────────────────────┐
                          │            FastAPI Backend                │
                          │  (Pydantic Validation · Error Handling)  │
                          └─────────────┬────────────────────────────┘
                                        │
                          ┌─────────────▼────────────────────────────┐
                          │         Input Sanitization                │
                          │  (SQL Injection · SSRF · Path Traversal) │
                          └─────────────┬────────────────────────────┘
                                        │
                          ┌─────────────▼────────────────────────────┐
                          │        NeMo Guardrails (Input)           │
                          │  (Jailbreak · Off-Topic Detection)      │
                          └─────────────┬────────────────────────────┘
                                        │
                    ┌───────────────────▼───────────────────┐
                    │         LangGraph Pipeline              │
                    │                                         │
                    │  ┌─────────┐    ┌──────────┐          │
                    │  │  Route   │───▶│ Retrieve │          │
                    │  │  Query   │    │          │          │
                    │  └─────────┘    └────┬─────┘          │
                    │       │              │                 │
                    │       │         ┌────▼─────┐          │
                    │       │         │  Rerank   │          │
                    │       │         └────┬─────┘          │
                    │       │              │                 │
                    │       │         ┌────▼─────┐          │
                    │       │         │ Generate  │          │
                    │       │         └────┬─────┘          │
                    │       │              │                 │
                    │       │         ┌────▼─────────┐      │
                    │       │         │ Hallucination │◀────┐│
                    │       │         │    Check      │     ││
                    │       │         └────┬─────────┘     ││
                    │       │              │ Pass          ││
                    │  ┌────▼─────┐  ┌────▼──────┐        ││
                    │  │Conversat-│  │ Out-of-   │        ││
                    │  │ional     │  │ Scope     │        ││
                    │  └──────────┘  └───────────┘        ││
                    └──────────────────────────────────────┘│
                                        │                    │
                          ┌─────────────▼────────────────┐  │
                          │     NeMo Guardrails (Output)  │  │
                          │     (Self-Check · Blocking)   │  │
                          └─────────────┬────────────────┘  │
                                        │                    │
                          ┌─────────────▼────────────────────┐
                          │       LangSmith + RAGAS          │
                          │  (Tracing · Feedback · Eval)     │
                          └──────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
            ┌─────────────┐   ┌──────────────┐   ┌──────────────┐
            │   ChromaDB   │   │  Query Cache  │   │  Embedding   │
            │  (Vectors)   │   │  (LRU + TTL)  │   │    Cache     │
            └─────────────┘   └──────────────┘   └──────────────┘
```

---

## ✨ Key Features

### 📥 Multi-Source Ingestion
- **Files** — PDF, DOCX, TXT, Markdown, CSV with automatic format detection
- **Web URLs** — HTML scraping with tag stripping and metadata extraction
- **REST APIs** — JSON endpoint ingestion for structured data sources
- **Raw Text** — Direct text input with custom source naming
- **Directory Scan** — Recursive batch ingestion with format-aware loading

### 🔍 Hybrid Retrieval
- **Dense + Sparse** — ChromaDB cosine similarity fused with BM25 lexical matching
- **Reciprocal Rank Fusion** — Configurable alpha-blending of dense and sparse scores
- **HyDE Expansion** — Hypothetical Document Embeddings improve query-to-document matching
- **Cross-Encoder Reranking** — `ms-marco-MiniLM-L-6-v2` with sigmoid-normalized scoring

### 🧠 Intelligent Generation
- **Query Routing** — Classifies queries as *Retrieval*, *Conversational*, or *Out-of-Scope*
- **Hallucination Detection** — LLM-based grounding check with one automated retry
- **"I Don't Know" / No Docs** — Refuses to answer when no relevant documents are found
- **Token-Budgeted Context** — Enforces max token limits with source attribution `[Source N]`

### 🛡 3-Layer Security
| Layer | Mechanism | Protects Against |
|-------|-----------|-----------------|
| **Input Sanitization** | Type/length validation, SQL injection detection, SSRF blocking, path traversal stripping | Injection attacks, SSRF, malicious uploads |
| **NeMo Guardrails (Input)** | Colang dialogue flows for greetings, off-topic, and jailbreak patterns | Prompt injection, abuse, off-topic queries |
| **NeMo Guardrails (Output)** | LLM self-check against safety policy (no harm, weapons, illegal content) | Harmful generated content |

### 📊 Full Observability
- **LangSmith Tracing** — Per-stage latency breakdown, run tracking, and session history
- **User Feedback** — Thumbs up/down with comments, integrated into LangSmith
- **RAGAS Evaluation** — Faithfulness, answer relevancy, context precision & recall metrics

### ⚡ Performance
- **Two-Layer Cache** — Query result cache (512 entries, 1h TTL) + embedding cache (2048 entries, 24h TTL)
- **Streaming** — Token-by-token WebSocket streaming for real-time responses
- **Batch Embedding** — Cached batch embed with retry logic for resilience

---

## 🧰 Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.11 |
| **API Framework** | FastAPI + Uvicorn (async REST + WebSocket) |
| **LLM** | Groq — LLaMA 3.3 70B Versatile |
| **Orchestration** | LangGraph (state machine pipeline) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` (384-dim) |
| **Vector Store** | ChromaDB (Docker, HNSW index, cosine similarity) |
| **Lexical Search** | Rank BM25 |
| **Reranker** | Cross-Encoder `ms-marco-MiniLM-L-6-v2` |
| **Guardrails** | NeMo Guardrails (Colang dialogue flows) |
| **Observability** | LangSmith + RAGAS |
| **Frontend** | Streamlit (chat UI, streaming, source cards) |
| **Caching** | In-memory LRU + TTL (thread-safe) |
| **Validation** | Pydantic v2 (settings, request/response models) |
| **Containerization** | Docker Compose (3 services, health checks) |
| **Package Manager** | UV |

---

## 📁 Project Structure

```
Enterprise_Rag/
├── config/
│   ├── settings.py                 # Pydantic Settings (central config from .env)
│   └── guardrails/
│       ├── config.yml              # NeMo Guardrails model + rail config
│       ├── prompts.yml             # Output self-check safety prompt
│       └── flows.co                # Colang flows (greeting, off-topic, jailbreak)
│
├── src/
│   ├── orchestrator.py             # Master orchestrator — wires all components
│   ├── api/
│   │   └── main.py                 # FastAPI app — REST + WebSocket endpoints
│   ├── ingestion/
│   │   ├── loaders.py              # Multi-source document loaders
│   │   ├── chunker.py              # Two-pass semantic chunking pipeline
│   │   └── vector_store.py         # ChromaDB manager + embedding cache
│   ├── retrieval/
│   │   ├── hybrid_retriever.py     # Hybrid BM25 + dense retrieval + HyDE
│   │   └── reranker.py             # Cross-encoder reranker + context builder
│   ├── generation/
│   │   └── rag_pipeline.py         # LangGraph pipeline (route → generate → verify)
│   ├── security/
│   │   └── guardrails.py           # NeMo Guardrails integration (input + output)
│   ├── observability/
│   │   ├── tracking.py             # LangSmith tracing + feedback
│   │   └── evaluator.py            # RAGAS evaluation (faithfulness, relevancy, etc.)
│   └── utils/
│       ├── cache.py                # Two-layer LRU/TTL cache (query + embedding)
│       └── sanitizer.py           # Input sanitization (SQLi, SSRF, file validation)
│
├── frontend/
│   └── app.py                       # Streamlit chat UI with streaming + source cards
│
├── scripts/
│   └── cli.py                       # CLI tool for ingestion and querying
│
├── docker/
│   ├── Dockerfile.api               # Multi-stage API image
│   ├── Dockerfile.frontend           # Frontend image
│   └── prometheus.yml                # Prometheus scrape config
│
├── tests/
│   ├── test_core.py                 # Chunker, loader, context builder tests
│   └── test_sanitizer.py            # Input sanitization tests
│
├── docker-compose.yml               # 3-service orchestration (ChromaDB + API + Frontend)
├── pyproject.toml                   # Project metadata + dependencies
├── requirements-api.txt             # Pinned API dependencies (for Docker)
├── requirements-frontend.txt         # Frontend dependencies
└── .env                             # Environment configuration (API keys, model params)
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- **Docker & Docker Compose** (for containerized deployment)
- **API Keys**: Groq, HuggingFace, and LangSmith

### 1. Clone the Repository

```bash
git clone https://github.com/DineshKumarVerma/Enterprise_Rag.git
cd Enterprise_Rag
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
# ── LLM ──────────────────────────────────────────────
GROQ_API_KEY="your-groq-api-key"
GROQ_MODEL=llama-3.3-70b-versatile

# ── Embeddings ────────────────────────────────────────
HF_API_KEY="your-huggingface-api-key"
EMBEDDING_MODEL=all-MiniLM-L6-v2

# ── Vector Store ──────────────────────────────────────
CHROMA_HOST=127.0.0.1
CHROMA_PORT=8001
CHROMA_COLLECTION=enterprise_rag

# ── LangSmith ─────────────────────────────────────────
LANGSMITH_API_KEY="your-langsmith-api-key"
LANGSMITH_PROJECT=enterprise-rag
LANGCHAIN_TRACING_V2=true
```

### 3. Deploy with Docker Compose

```bash
docker compose up --build
```

This starts three services:
| Service | Port | Description |
|---------|------|-------------|
| **ChromaDB** | `8001` | Vector database with persistent storage |
| **API** | `8000` | FastAPI backend with REST + WebSocket |
| **Frontend** | `8501` | Streamlit chat interface |

Once healthy, open **http://localhost:8501** for the chat UI or **http://localhost:8000/docs** for the interactive API docs.

### 3. Alternative: Run Locally

```bash
# Install UV package manager
pip install uv

# Create virtual environment and install dependencies
uv venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate    # Linux/macOS

uv pip install -e ".[dev,frontend,eval]"

# Start ChromaDB (separate terminal)
docker run -d -p 8001:8000 chromadb/chroma:0.5.15

# Start the API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Start the Frontend (separate terminal)
streamlit run frontend/app.py
```

---

## 📡 API Reference

All endpoints are available at `http://localhost:8000`. Full interactive docs at `/docs` (Swagger UI).

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check — returns status, app name, environment |
| `GET` | `/stats` | Vector store stats, cache stats, and settings summary |
| `POST` | `/query` | RAG query — returns answer, sources, query type, grounding status, latencies |
| `WS` | `/ws/query` | Streaming query — token-by-token via WebSocket |
| `POST` | `/ingest/file` | Upload and ingest a file (PDF, DOCX, TXT, MD, CSV) |
| `POST` | `/ingest/url` | Ingest content from a web URL |
| `POST` | `/ingest/text` | Ingest raw text with a source name |
| `POST` | `/feedback` | Submit user feedback (score 0–1 + comment) |
| `POST` | `/evaluate` | RAGAS evaluation — faithfulness, relevancy, precision, recall |
| `POST` | `/cache/clear` | Clear the query result cache |

### Example Queries

**Query the RAG system:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the company remote work policy?", "use_hyde": true}'
```

**Ingest a file:**
```bash
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@employee_handbook.pdf"
```

**Ingest a URL:**
```bash
curl -X POST http://localhost:8000/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/policy-page"}'
```

**Submit feedback:**
```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"run_id": "abc123", "score": 0.9, "comment": "Great answer!"}'
```

---

## 🖥 CLI Tool

The project includes a command-line interface for quick operations:

```bash
# Ingest a file
python -m scripts.cli ingest --file document.pdf

# Ingest a URL
python -m scripts.cli ingest --url https://example.com/docs

# Ingest a directory (recursive)
python -m scripts.cli ingest --dir ./data/

# Query the system
python -m scripts.cli query "What is the return policy?"

# Query without HyDE expansion
python -m scripts.cli query "What is the return policy?" --no-hyde

# View stats
python -m scripts.cli stats
```

---

## ⚙ Configuration

All settings are managed through `config/settings.py` (Pydantic BaseSettings) and loaded from `.env`:

| Setting | Default | Description |
|---------|---------|-------------|
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | LLM for generation, routing, HyDE, hallucination check |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer for embeddings |
| `CHROMA_HOST` / `CHROMA_PORT` | `127.0.0.1:8001` | ChromaDB connection |
| `TOP_K_RETRIEVAL` | `20` | Candidates retrieved before reranking |
| `TOP_K_RERANK` | `5` | Top documents after cross-encoder reranking |
| `RELEVANCE_THRESHOLD` | `0.35` | Minimum relevance score to include in context |
| `HYBRID_ALPHA` | `0.5` | Dense vs BM25 fusion weight (0=BM25 only, 1=dense only) |
| `MAX_CONTEXT_TOKENS` | `8000` | Token budget for context window |
| `TEMPERATURE` | `0.1` | LLM generation temperature |

---

## 🧪 Testing

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_sanitizer.py
```

**Test Coverage:**
- **`test_core.py`** — Chunking pipeline, document loading, context builder logic
- **`test_sanitizer.py`** — SQL injection detection, SSRF prevention, file validation, path traversal

---

## 🔒 Security Design

The security model follows a **defense-in-depth** approach with three independent layers:

```
┌──────────────────────────────────────────────┐
│  Layer 1: Input Sanitization                 │
│  ┌──────────┐ ┌───────┐ ┌──────────────────┐ │
│  │   SQLi   │ │  SSRF │ │  File Validation │ │
│  │ Detection│ │ Block │ │  (type/size/path)│ │
│  └──────────┘ └───────┘ └──────────────────┘ │
├──────────────────────────────────────────────┤
│  Layer 2: NeMo Guardrails (Input)            │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│  │Jailbreak │ │Off-Topic │ │  Greeting    │  │
│  │ Detection│ │ Blocking │ │  Handling   │  │
│  └──────────┘ └──────────┘ └──────────────┘  │
├──────────────────────────────────────────────┤
│  Layer 3: NeMo Guardrails (Output)           │
│  ┌──────────────────────────────────────────┐ │
│  │  Self-Check: blocks harm, weapons,       │ │
│  │  illegal content in generated responses  │ │
│  └──────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
```

---

## 📈 Observability

### LangSmith Tracing

Every query is traced through LangSmith with per-stage latency:

- **Retrieval** — dense search + BM25 + fusion time
- **Reranking** — cross-encoder scoring time
- **Generation** — LLM inference time
- **Guardrails** — input + output check time

### RAGAS Evaluation

Evaluate RAG quality with industry-standard metrics:

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the policy?",
    "answer": "Employees may work remotely...",
    "contexts": ["The remote work policy states..."],
    "ground_truth": "Up to 3 days per week with approval"
  }'
```

Returns: **faithfulness**, **answer relevancy**, **context precision**, **context recall**

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](./LICENSE) for details.

```
MIT License — Copyright (c) 2026 Dinesh Kumar Verma
```

---

## 👤 Author

**Dinesh Kumar Verma**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/dinesh-kumar-verma)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?logo=github&logoColor=white)](https://github.com/DineshKumarVerma)

---

<div align="center">

**If you found this project helpful, please consider giving it a ⭐!**

</div>
