"""
embeddings.py
-------------
Loads all company chunks from JSON, generates vector embeddings using
OpenAI text-embedding-3-small, and saves a FAISS index to disk.

Model: text-embedding-3-small
  - 1536 dimensions
  - Fast and cheap (~$0.02 per 1M tokens)
  - No local GPU/RAM requirements

Usage:
    python -m src.embeddings.embeddings
"""

import json
import os
from pathlib import Path
import pickle

from dotenv import load_dotenv
import faiss
import numpy as np
from openai import OpenAI
from tqdm import tqdm

from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")
INDEX_DIR = Path("data/index")

MODEL_NAME = "text-embedding-3-small"
BATCH_SIZE = 512
COMPANIES = ["tesla", "apple", "microsoft"]


# ── Data loading ──────────────────────────────────────────────────────────────


def load_all_chunks() -> list[dict]:
    all_chunks = []
    doc_id = 0

    for company in COMPANIES:
        path = PROCESSED_DIR / f"{company}_chunks.json"
        if not path.exists():
            logger.warning("Missing chunks file: %s — run processor.py first", path)
            continue
        chunks = json.loads(path.read_text(encoding="utf-8"))
        for chunk in chunks:
            chunk["doc_id"] = doc_id
            doc_id += 1
        all_chunks.extend(chunks)
        logger.info("Loaded %d chunks ← %s", len(chunks), company)

    return all_chunks


# ── Embedding ─────────────────────────────────────────────────────────────────


def generate_embeddings(chunks: list[dict], client: OpenAI) -> np.ndarray:
    """Encode all chunks in batches via OpenAI API. Returns float32 (N, 1536)."""
    texts = [chunk["text"] for chunk in chunks]
    all_embeddings = []

    logger.info("Encoding %d chunks in batches of %d...", len(texts), BATCH_SIZE)

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
        batch = texts[i : i + BATCH_SIZE]
        response = client.embeddings.create(model=MODEL_NAME, input=batch)
        all_embeddings.extend([item.embedding for item in response.data])

    embeddings = np.array(all_embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)  # L2-norm → inner product == cosine similarity
    logger.info("Embeddings shape: %s", embeddings.shape)
    return embeddings


# ── FAISS index ───────────────────────────────────────────────────────────────


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("FAISS index built — %d vectors (dim=%d)", index.ntotal, dim)
    return index


# ── Persistence ───────────────────────────────────────────────────────────────


def save_artifacts(index: faiss.IndexFlatIP, chunks: list[dict]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(chunks, f)
    logger.info("Saved FAISS index + metadata (%d records)", len(chunks))


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise OSError("OPENAI_API_KEY not set in .env")

    client = OpenAI(api_key=api_key)

    logger.info("=== STEP 1 — Loading chunks ===")
    chunks = load_all_chunks()
    if not chunks:
        logger.error("No chunks found. Aborting.")
        return

    logger.info("=== STEP 2 — Generating embeddings via OpenAI (%s) ===", MODEL_NAME)
    embeddings = generate_embeddings(chunks, client)

    logger.info("=== STEP 3 — Building & saving FAISS index ===")
    index = build_faiss_index(embeddings)
    save_artifacts(index, chunks)
    logger.info("Done. Run retriever.py to test a query.")


if __name__ == "__main__":
    main()
