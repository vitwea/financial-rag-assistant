"""
bm25_index.py
-------------
BM25 lexical search over the chunk corpus.

BM25 complements semantic (FAISS) search for queries that contain
exact keywords, numbers, or named entities that embeddings struggle with.

Examples where BM25 outperforms embeddings:
  - "iPhone revenue 2024"
  - "Azure grew 29%"
  - "EBITDA margin Tesla"

The index is built in memory at startup — no extra disk artefact needed.

Usage:
    from src.retrieval.bm25_index import build_bm25_index, bm25_search
"""

import re

from rank_bm25 import BM25Okapi

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Tokeniser ─────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> list[str]:
    """
    Financial-aware tokeniser:
      - Lowercase
      - Keeps numbers and percentages intact ("29%", "$42B")
      - Removes punctuation except inside numbers
    """
    text   = text.lower()
    text   = re.sub(r"[^\w\s%$\.\-]", " ", text)
    tokens = [t for t in text.split() if re.search(r"\w", t)]
    return tokens


# ── Index builder ─────────────────────────────────────────────────────────────

def build_bm25_index(metadata: list[dict]) -> BM25Okapi:
    """Build a BM25Okapi index from the chunk corpus."""
    logger.info("Building BM25 index over %d chunks...", len(metadata))
    corpus = [_tokenise(chunk["text"]) for chunk in metadata]
    index  = BM25Okapi(corpus)
    logger.info("BM25 index ready")
    return index


# ── Search ────────────────────────────────────────────────────────────────────

def bm25_search(
    query:          str,
    bm25:           BM25Okapi,
    metadata:       list[dict],
    top_k:          int = 20,
    company_filter: str | None = None,
    year_filter:    int | None = None,
) -> list[dict]:
    """
    BM25 lexical search with optional company/year filters.
    Returns up to top_k results with a 'bm25_score' field attached.
    """
    tokens = _tokenise(query)
    scores = bm25.get_scores(tokens)

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in ranked:
        if score <= 0:
            break   # no lexical match

        chunk = dict(metadata[idx])

        if company_filter and chunk.get("company") != company_filter:
            continue
        if year_filter and chunk.get("year") != year_filter:
            continue

        chunk["bm25_score"] = float(score)
        results.append(chunk)

        if len(results) == top_k:
            break

    logger.debug("BM25: %d results (company=%s year=%s)", len(results), company_filter, year_filter)
    return results
