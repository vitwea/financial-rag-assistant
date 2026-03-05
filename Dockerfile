FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── 1. CPU-only PyTorch ───────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    "torch==2.3.0+cpu" \
    "torchvision==0.18.0+cpu" \
    --index-url https://download.pytorch.org/whl/cpu

# ── 2. sentence-transformers without deps ─────────────────────────────────────
RUN pip install --no-cache-dir --no-deps sentence-transformers==2.7.0 && \
    pip install --no-cache-dir \
        transformers>=4.39.0 \
        huggingface-hub>=0.23.0 \
        tokenizers>=0.19.0 \
        Pillow>=9.0.0 \
        scikit-learn \
        scipy

# ── 3. Everything else ────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    requests>=2.31.0 \
    python-dotenv>=1.0.0 \
    pypdf>=4.0.0 \
    beautifulsoup4>=4.12.0 \
    faiss-cpu>=1.8.0 \
    numpy>=1.26.0 \
    tqdm>=4.66.0 \
    openai>=1.30.0 \
    anthropic>=0.28.0 \
    cohere>=5.5.0 \
    fastapi>=0.111.0 \
    "uvicorn[standard]>=0.29.0" \
    pydantic>=2.7.0 \
    streamlit>=1.35.0 \
    pandas>=2.2.0

# ── 4. Copy app ───────────────────────────────────────────────────────────────
COPY src/ ./src/
COPY data/index/ ./data/index/
COPY data/processed/ ./data/processed/
COPY start.sh .

RUN mkdir -p logs && chmod +x start.sh

# Unset STREAMLIT_SERVER_PORT so Railway's auto-injection doesn't interfere
ENV STREAMLIT_SERVER_PORT=""

EXPOSE 8501

CMD ["/bin/bash", "start.sh"]