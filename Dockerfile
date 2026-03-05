FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Step 1: Install PyTorch CPU-only FIRST (prevents CUDA version being pulled) ──
# This alone reduces image size from ~8GB to ~2GB
RUN pip install --no-cache-dir \
    torch==2.3.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ── Step 2: Install remaining dependencies ────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Step 3: Copy source and data ──────────────────────────────────────────────
COPY src/ ./src/
COPY data/index/ ./data/index/
COPY data/processed/ ./data/processed/

RUN mkdir -p logs

EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
