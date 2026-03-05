# 📊 Financial RAG Assistant

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=for-the-badge&logo=openai&logoColor=white"/>
  <img src="https://img.shields.io/badge/Anthropic-Claude%20Judge-D97757?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/FAISS-Vector%20Search-00BFFF?style=for-the-badge"/>
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
Retriever ────────── FAISS vector search → Cohere rerank → balanced selection
    │
    ▼
LLM generation ───── GPT-4o-mini · grounded system prompt · citation rules
    │
    ▼
Post-guardrails ───── hallucination detection · citation check
    │
    ▼
Evaluator (opt.) ──── Claude-as-judge: grounding · relevance · faithfulness · completeness
    │
    ▼
Response ─────────── answer + sources + quality scores + latency
```

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector store | FAISS (CPU, cosine similarity) |
| Reranking | Cohere Rerank v3 |
| Generation | OpenAI GPT-4o-mini |
| Evaluation judge | Anthropic Claude Haiku |
| Guardrails | Custom pre/post pipeline (`src/evaluation/guardrails.py`) |
| REST API | FastAPI + Pydantic v2 |
| UI | Streamlit |
| Containerisation | Docker + docker-compose |
| CI | GitHub Actions (lint → test → docker build) |
| Linting | Ruff |
| Testing | pytest + pytest-cov (all external calls mocked) |

---

## Project structure

```
financial-rag-assistant/
├── src/
│   ├── ingestion/          # SEC EDGAR downloader · HTML/PDF extractor · chunker
│   ├── embeddings/         # sentence-transformers → FAISS index builder
│   ├── retrieval/          # FAISS search · Cohere rerank · balanced selection
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
COHERE_API_KEY=...          # optional — falls back to FAISS order if unset
```

### 3. Build the index

```bash
# Download 10-K filings from SEC EDGAR (2022–2025)
python -m src.ingestion.downloader

# Clean and chunk the documents
python -m src.ingestion.processor

# Embed and build the FAISS index
python -m src.embeddings.embeddings
```

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
  "answer": "Tesla faces several key risk factors:\n- Macroeconomic risks including inflation [Tesla | pages 10–11].\n- Supply chain disruptions and semiconductor shortages [Tesla | pages 16–17].\n- Key personnel dependency on Elon Musk [Tesla | pages 12–13].",
  "sources": "**Sources used:**\n  • Tesla 10-K | pages 10–11 | tesla_10k_2024.htm\n  • Tesla 10-K | pages 16–17 | tesla_10k_2024.htm",
  "source_details": [
    {"company": "tesla", "source": "tesla_10k_2024.htm", "start_page": 10, "end_page": 11, "score": 0.9132}
  ],
  "latency_ms": 1243
}
```

---

## Evaluation

The pipeline includes an **LLM-as-a-judge** system using `claude-haiku-4-5` to score every response:

| Dimension | Definition | Pass threshold |
|-----------|-----------|----------------|
| **Grounding** | Every claim backed by retrieved passages | ≥ 0.70 |
| **Relevance** | Answer directly addresses the question | ≥ 0.70 |
| **Faithfulness** | No information added beyond the context | ≥ 0.80 |
| **Completeness** | All answerable aspects are covered | ≥ 0.60 |

Run the full benchmark (works without API keys using simulated data):

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
| `OPENAI_API_KEY` | ✅ | GPT-4o-mini for answer generation |
| `ANTHROPIC_API_KEY` | ✅ | Claude judge for response evaluation |
| `COHERE_API_KEY` | ⬜ | Reranking — falls back to FAISS order if not set |
