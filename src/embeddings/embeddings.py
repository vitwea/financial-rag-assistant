"""
embeddings.py
-------------
Loads all company chunks from JSON, generates vector embeddings using a
local sentence-transformers model (no API cost), and saves a FAISS index
to disk for fast similarity search.

Model: BAAI/bge-large-en-v1.5
  - Top-ranked on the MTEB leaderboard
  - Strong semantic understanding for financial text
  - Runs fully local, no API key required

Usage:
    python src/embeddings/embeddings.py
"""

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")
INDEX_DIR     = Path("data/index")

MODEL_NAME  = "BAAI/bge-large-en-v1.5"
BATCH_SIZE  = 32
COMPANIES   = ["tesla", "apple", "microsoft"]

BGE_PREFIX = "Represent this sentence for searching relevant passages: "


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_chunks() -> list[dict]:
    """Load chunk records for all companies and attach a global doc_id."""
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

def generate_embeddings(chunks: list[dict],
                        model: SentenceTransformer) -> np.ndarray:
    """Encode all chunk texts in batches. Returns float32 array (N, dim)."""
    texts = [BGE_PREFIX + chunk["text"] for chunk in chunks]

    logger.info("Encoding %d chunks in batches of %d...", len(texts), BATCH_SIZE)
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    logger.info("Embeddings shape: %s", embeddings.shape)
    return embeddings.astype("float32")


# ── FAISS index ───────────────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS IndexFlatIP (inner product) index.
    L2-normalised embeddings make inner product == cosine similarity.
    """
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("FAISS index built — %d vectors (dim=%d)", index.ntotal, dim)
    return index


# ── Persistence ───────────────────────────────────────────────────────────────

def save_artifacts(index: faiss.IndexFlatIP, chunks: list[dict]) -> None:
    """Save the FAISS index and chunk metadata to disk."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    index_path    = INDEX_DIR / "faiss.index"
    metadata_path = INDEX_DIR / "metadata.pkl"

    faiss.write_index(index, str(index_path))
    logger.info("Saved FAISS index → %s", index_path)

    with open(metadata_path, "wb") as f:
        pickle.dump(chunks, f)
    logger.info("Saved metadata → %s (%d records)", metadata_path, len(chunks))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("=== STEP 1 — Loading chunks ===")
    chunks = load_all_chunks()

    if not chunks:
        logger.error("No chunks found. Aborting.")
        return

    logger.info("Total chunks: %d", len(chunks))

    logger.info("=== STEP 2 — Loading embedding model: %s ===", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    logger.info("=== STEP 3 — Generating embeddings ===")
    embeddings = generate_embeddings(chunks, model)

    logger.info("=== STEP 4 — Building & saving FAISS index ===")
    index = build_faiss_index(embeddings)
    save_artifacts(index, chunks)

    logger.info("Done. Run retriever.py to test a query.")


if __name__ == "__main__":
    main()