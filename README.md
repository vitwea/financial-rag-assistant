# 📊 Financial RAG Assistant

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=for-the-badge&logo=openai&logoColor=white"/>
  <img src="https://img.shields.io/badge/Anthropic-Claude%20Judge-D97757?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/FAISS%20%2B%20BM25-Hybrid%20Search-00BFFF?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Cohere-Rerank-6B4FBB?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/FastAPI-REST%20API-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-Chat%20UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/SEC%2010--K-Tesla%20%7C%20Apple%20%7C%20Microsoft-E31937?style=for-the-badge"/>
</p>

<p align="center">
  <img src="https://github.com/vitwea/financial-rag-assistant/actions/workflows/ci.yml/badge.svg"/>
</p>

<p align="center">
  <em>Ask complex questions about Tesla, Apple and Microsoft annual reports — every answer is grounded in real 10-K passages with page-level citations.</em>
</p>

---

## What it does

The Financial RAG Assistant lets you query SEC 10-K filings for **Tesla, Apple and Microsoft** (2022–2025) using natural language. Rather than hallucinating answers, it retrieves the relevant passages first and only responds based on what is actually in the documents — with citations down to the exact page number.

Every response goes through a **multi-stage guardrail system** and can be automatically scored by a Claude judge across four quality dimensions (grounding, relevance, faithfulness, completeness).

---

## Architecture

```
User query
    │
    ▼
Pre-guardrails ──── topic check · context quality check
    │
    ▼
Query expander ───── company-aware query rewriting
    │
    ▼
Hybrid retriever ─── FAISS semantic search ─┐
                     BM25 lexical search  ───┤→ Reciprocal Rank Fusion → Cohere rerank
                                             └── balanced selection (comparison queries)
    │
    ▼
LLM generation ───── GPT-4o-mini · grounded system prompt · citation rules
    │
    ▼
Post-guardrails ───── hallucination detection · citation check
    │
    ▼
Evaluator (opt.) ──── Claude-as-judge (2-run average): grounding · relevance · faithfulness · completeness
    │
    ▼
Response ─────────── answer + sources + quality scores + latency
```

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Embeddings | OpenAI `text-embedding-3-small` (1536-dim, via API) |
| Vector store | FAISS (CPU, cosine similarity via inner product) |
| Lexical search | BM25 (`rank-bm25`, financial-aware tokeniser) |
| Rank fusion | Reciprocal Rank Fusion (RRF, deterministic) |
| Reranking | Cohere Rerank v3 |
| Generation | OpenAI GPT-4o-mini |
| Evaluation judge | Anthropic Claude Haiku (2-run averaged scoring) |
| Guardrails | Custom pre/post pipeline (`src/evaluation/guardrails.py`) |
| REST API | FastAPI + Pydantic v2 |
| UI | Streamlit (chat-style, no sidebar) |
| Containerisation | Docker + docker-compose (no PyTorch — ~1.2 GB lighter) |
| CI | GitHub Actions (lint → test → docker build) |
| Linting | Ruff |
| Testing | pytest + pytest-cov (all external calls mocked) |

---

## Project structure

```
financial-rag-assistant/
├── src/
│   ├── ingestion/          # SEC EDGAR downloader · HTML/PDF extractor · chunker
│   ├── embeddings/         # OpenAI embeddings → FAISS index builder
│   ├── retrieval/
│   │   ├── bm25_index.py   # BM25 lexical index (in-memory, built at startup)
│   │   ├── retriever.py    # Hybrid search: FAISS + BM25 → RRF → Cohere rerank
│   │   └── query_expander.py
│   ├── pipeline/           # RAG orchestration: guardrails → retrieve → generate
│   ├── evaluation/         # Claude judge · guardrail checks · EvaluationResult
│   ├── api/                # FastAPI endpoints
│   └── utils/              # Centralised structured logger
├── tests/
│   ├── test_pipeline.py    # build_prompt · format_sources · RAGPipeline · API
│   ├── test_evaluation.py  # guardrails · EvaluationResult · RAGEvaluator (mocked)
│   └── test_ingestion.py   # clean_text · chunker · SEC downloader (mocked)
├── notebooks/
│   ├── 01_ingestion_analysis.ipynb    # chunk size distribution · text quality
│   ├── 02_retrieval_analysis.ipynb    # UMAP · FAISS vs Cohere rerank comparison
│   └── 03_evaluation_benchmark.ipynb # radar · heatmap · pass-rate by category
├── app.py                  # Streamlit chat UI
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml          # ruff · pytest · coverage config
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/vitwea/financial-rag-assistant.git
cd financial-rag-assistant
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
COHERE_API_KEY=...          # optional — falls back to RRF order if unset
```

### 3. Build the index

```bash
# Download 10-K filings from SEC EDGAR (2022–2025)
python -m src.ingestion.downloader

# Clean and chunk the documents
python -m src.ingestion.processor

# Embed via OpenAI API and build the FAISS index
# (uses text-embedding-3-small — requires OPENAI_API_KEY)
python -m src.embeddings.embeddings
```

> The BM25 index is built automatically in memory at startup — no extra step required.

### 4. Run

**Streamlit UI**
```bash
streamlit run app.py
# → http://localhost:8501
```

**FastAPI** (with interactive docs)
```bash
uvicorn src.api.api:app --reload
# → http://localhost:8000/docs
```

**Docker**
```bash
docker-compose up
# UI  → http://localhost:8501
# API → http://localhost:8000
```

---

## Retrieval

The retriever uses a three-stage hybrid pipeline:

**Stage 1 — Hybrid search**
FAISS (semantic) and BM25 (lexical) run in parallel. BM25 excels where embeddings struggle: exact numbers ("Azure grew 29%"), product names ("iPhone revenue"), and financial acronyms ("EBITDA margin"). Results are merged via **Reciprocal Rank Fusion** — rank-based fusion that is robust to score scale differences.

**Stage 2 — Cohere rerank**
A cross-encoder rescores the merged candidate pool against the original query. Falls back to RRF order if `COHERE_API_KEY` is not set.

**Stage 3 — Balanced selection (comparison queries)**
For queries comparing companies or time periods, the retriever runs per-entity searches and guarantees at least one chunk per company/year in the final context — preventing dominant sources from crowding out others.

**Query routing:**

| Query type | Example | Strategy |
|---|---|---|
| Exact filter | Company or year set in UI | Hybrid search with filter |
| Temporal | "How has Azure evolved over the years?" | Per-year balanced retrieval |
| Comparison | "Compare R&D across all three companies" | Per-company balanced retrieval |
| Standard | "What are Tesla's risk factors?" | Global hybrid search |

---

## API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ask` | Answer a financial question |
| `GET` | `/health` | Index status and chunk count |
| `GET` | `/companies` | List of available companies |

**Request:**

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are Tesla main risk factors?",
    "company_filter": "tesla",
    "year_filter": 2024
  }'
```

**Response:**

```json
{
  "query": "What are Tesla main risk factors?",
  "answer": "Tesla faces several key risk factors:\n- Key personnel dependency on Elon Musk [Tesla | pages 27–29].\n- Supply chain and manufacturing scale risks [Tesla | pages 18–19].\n- Cybersecurity vulnerabilities [Tesla | pages 44–46].",
  "sources": "**Sources used:**\n  • Tesla 10-K | pages 27–29 | tesla_10k_2024.htm",
  "source_details": [
    {"company": "tesla", "source": "tesla_10k_2024.htm", "start_page": 27, "end_page": 29, "score": 0.9132}
  ],
  "latency_ms": 1243
}
```

---

## Evaluation

The pipeline includes an **LLM-as-a-judge** system using `claude-haiku-4-5` that scores every response across four dimensions. To reduce variance from LLM non-determinism, scores are **averaged over 2 independent judge calls** (±0.05 stability).

Thresholds are calibrated for financial document synthesis, where paraphrasing and cross-passage synthesis are expected behaviours — not faithfulness violations:

| Dimension | Definition | Pass threshold |
|-----------|-----------|----------------|
| **Grounding** | Claims traceable to retrieved passages (paraphrases count) | ≥ 0.55 |
| **Relevance** | Answer directly addresses the question | ≥ 0.65 |
| **Faithfulness** | No facts invented beyond the context | ≥ 0.55 |
| **Completeness** | All answerable aspects are covered | ≥ 0.50 |

Run the full benchmark:

```bash
jupyter notebook notebooks/03_evaluation_benchmark.ipynb
```

The notebook produces a **radar chart** per company, a **score heatmap** across all queries, and a **pass-rate breakdown by category** — and exports results to `data/eval_summary.json` for regression tracking over time.

---

## Guardrails

| Stage | Check | Behaviour |
|-------|-------|-----------|
| Pre | Topic check | Rejects off-topic queries (e.g. weather, stock prices) |
| Pre | Context quality | Blocks if retrieved passage scores are too low |
| Post | Hallucination detection | Flags hedge phrases not grounded in the context |
| Post | Citation check | Verifies every answer includes `[Company \| pages X–Y]` |

---

## Tests

All external calls (OpenAI, Anthropic, Cohere, FAISS) are mocked — the full suite runs in seconds without any API keys.

```bash
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

CI runs automatically on every push and pull request: **lint (ruff) → tests (pytest) → Docker build**.

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | GPT-4o-mini for generation + `text-embedding-3-small` for indexing |
| `ANTHROPIC_API_KEY` | ✅ | Claude judge for response evaluation |
| `COHERE_API_KEY` | ⬜ | Reranking — falls back to RRF order if not set |
