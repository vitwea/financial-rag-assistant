FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Pin PyTorch to CPU-only BEFORE anything else ──────────────────────────────
# Must happen before requirements.txt so sentence-transformers doesn't pull CUDA
RUN pip install --no-cache-dir \
    "torch==2.3.0+cpu" \
    "torchvision==0.18.0+cpu" \
    --index-url https://download.pytorch.org/whl/cpu

# ── Install sentence-transformers without letting it upgrade torch ─────────────
RUN pip install --no-cache-dir --no-deps sentence-transformers==2.7.0
RUN pip install --no-cache-dir \
    transformers>=4.39.0 \
    huggingface-hub>=0.23.0 \
    tokenizers>=0.19.0 \
    Pillow>=9.0.0 \
    scikit-learn \
    scipy \
    tqdm

# ── Install remaining requirements (torch already pinned, won't be touched) ───
COPY requirements.txt .
# Remove sentence-transformers from requirements since already installed
RUN grep -v "sentence-transformers" requirements.txt > requirements_filtered.txt && \
    pip install --no-cache-dir -r requirements_filtered.txt

# ── Copy app ──────────────────────────────────────────────────────────────────
COPY src/ ./src/
COPY data/index/ ./data/index/
COPY data/processed/ ./data/processed/

RUN mkdir -p logs

EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]