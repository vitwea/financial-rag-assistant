FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependencies ──────────────────────────────────────────────────────────────
# No PyTorch or sentence-transformers — embeddings via OpenAI API
RUN pip install --no-cache-dir \
    requests>=2.31.0 \
    python-dotenv>=1.0.0 \
    pypdf>=4.0.0 \
    beautifulsoup4>=4.12.0 \
    faiss-cpu>=1.8.0 \
    numpy>=1.26.0 \
    tqdm>=4.66.0 \
    rank-bm25>=0.2.2 \
    openai>=1.30.0 \
    anthropic>=0.28.0 \
    cohere>=5.5.0 \
    fastapi>=0.111.0 \
    "uvicorn[standard]>=0.29.0" \
    pydantic>=2.7.0 \
    streamlit>=1.35.0 \
    pandas>=2.2.0

# ── Copy app ──────────────────────────────────────────────────────────────────
COPY app.py .
COPY src/ ./src/
COPY data/index/ ./data/index/
COPY data/processed/ ./data/processed/

RUN mkdir -p logs

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]