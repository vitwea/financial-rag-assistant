"""
retriever.py
------------
Two-stage retrieval pipeline with company-aware AND year-aware balancing.

  Stage 1 — Vector search (FAISS)
      Supports filtering by company and/or year.
      For comparison queries: per-entity retrieval guarantees balanced pools.

  Stage 2 — Re-ranking (Cohere)
      Cross-encoder scores all candidates against the original query.

  Stage 3 — Balanced selection (comparison queries only)
      Guarantees representation from all requested companies/years.

Query routing:
  - company_filter set         → single company, all years
  - year_filter set            → single year, all companies
  - Temporal comparison query  → per-year balanced retrieval
  - Company comparison query   → per-company balanced retrieval
  - Standard query             → global FAISS top-k

Usage:
    python -m src.retrieval.retriever
"""

from functools import lru_cache
import os
from pathlib import Path
import pickle
import re

import cohere
from dotenv import load_dotenv
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrieval.query_expander import expand_query
from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

INDEX_DIR  = Path("data/index")
MODEL_NAME = "BAAI/bge-large-en-v1.5"

FAISS_TOP_K       = 20
RERANK_TOP_K      = 5
FAISS_PER_ENTITY  = 8     # chunks fetched per company OR per year in comparison mode
MIN_PER_ENTITY    = 1     # guaranteed slots per entity in final output

COHERE_RERANK_MODEL = "rerank-english-v3.0"


# ── Cached model loader ───────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load the BGE model once and reuse across all queries."""
    logger.info("Loading embedding model (once): %s", MODEL_NAME)
    return SentenceTransformer(MODEL_NAME)
BGE_QUERY_PREFIX    = "Represent this question for searching relevant passages: "

ALL_COMPANIES = ["tesla", "apple", "microsoft"]

COMPARISON_KEYWORDS = re.compile(
    r"\b(compare|comparison|versus|vs\.?|between|all (three|companies)|"
    r"each company|tesla.+apple|apple.+microsoft|tesla.+microsoft|"
    r"microsoft.+apple|apple.+tesla|microsoft.+tesla)\b",
    re.IGNORECASE,
)

# Patterns to detect temporal comparison queries
TEMPORAL_KEYWORDS = re.compile(
    r"\b(evolution|evolve|evolved|trend|over the years|over time|"
    r"year.over.year|yoy|historically|history|changed|change|"
    r"progress|progression|compared to \d{4}|\d{4}.+(vs|versus|compared)|\d{4}.+\d{4})\b",
    re.IGNORECASE,
)


# ── Index loading ─────────────────────────────────────────────────────────────

def load_index() -> tuple[faiss.IndexFlatIP, list[dict]]:
    index_path    = INDEX_DIR / "faiss.index"
    metadata_path = INDEX_DIR / "metadata.pkl"

    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Index not found. Run src/embeddings/embeddings.py first."
        )

    index = faiss.read_index(str(index_path))
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    logger.info("Index loaded: %d vectors", index.ntotal)
    logger.info("Metadata: %d chunks", len(metadata))
    return index, metadata


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    prefixed = BGE_QUERY_PREFIX + query
    vec      = model.encode([prefixed], normalize_embeddings=True)
    return vec.astype("float32")


# ── FAISS search ──────────────────────────────────────────────────────────────

def faiss_search(
    query:          str,
    index:          faiss.IndexFlatIP,
    metadata:       list[dict],
    top_k:          int = FAISS_TOP_K,
    company_filter: str | None = None,
    year_filter:    int | None = None,
) -> list[dict]:
    """
    Vector search with optional company and/or year filters.
    Returns top_k results after applying filters.
    """
    model = get_embedding_model()
    vec   = embed_query(query, model)

    # Search a larger pool when filtering (to ensure enough results after filter)
    k = min(index.ntotal, top_k * 10 if (company_filter or year_filter) else top_k)
    scores, indices = index.search(vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0], strict=False):
        if idx < 0 or idx >= len(metadata):
            continue
        chunk = dict(metadata[idx])

        # Apply filters
        if company_filter and chunk.get("company") != company_filter:
            continue
        if year_filter and chunk.get("year") != year_filter:
            continue

        chunk["faiss_score"] = float(score)
        results.append(chunk)

        if len(results) == top_k:
            break

    active_filter = f"company={company_filter}" if company_filter else ""
    if year_filter:
        active_filter += f" year={year_filter}"
    active_filter = active_filter.strip() or "None"

    logger.debug("FAISS: %d candidates (filter=%s)", len(results), active_filter)
    return results


def faiss_search_per_entity(
    query:           str,
    index:           faiss.IndexFlatIP,
    metadata:        list[dict],
    per_entity:      int = FAISS_PER_ENTITY,
    target_companies: list[str] | None = None,
    target_years:     list[int] | None = None,
    company_context:  str | None = None,
) -> list[dict]:
    """
    Retrieve chunks per company OR per year to guarantee balanced pools.

    - If target_years is set: one retrieval per year (optionally filtered to company_context)
    - If target_companies is set: one retrieval per company (across all years)
    """
    all_candidates: list[dict] = []
    seen_ids: set[tuple]       = set()

    model = get_embedding_model()

    if target_years:
        # Temporal comparison: retrieve per year
        for year in target_years:
            expanded = expand_query(query, company_context)
            vec      = embed_query(expanded, model)

            k = min(index.ntotal, per_entity * 15)
            scores, indices = index.search(vec, k)

            year_chunks = []
            for score, idx in zip(scores[0], indices[0], strict=False):
                if idx < 0 or idx >= len(metadata):
                    continue
                chunk = dict(metadata[idx])
                if chunk.get("year") != year:
                    continue
                # If a company context is set, only include chunks from that company
                if company_context and chunk.get("company") != company_context:
                    continue
                uid = (chunk["company"], chunk.get("year"), chunk["chunk_id"])
                if uid in seen_ids:
                    continue
                seen_ids.add(uid)
                chunk["faiss_score"] = float(score)
                year_chunks.append(chunk)
                if len(year_chunks) == per_entity:
                    break

            logger.debug("Per-year: %d chunks from year %d (company=%s)", len(year_chunks), year, company_context or "all")
            all_candidates.extend(year_chunks)

    elif target_companies:
        # Company comparison: retrieve per company
        for company in target_companies:
            expanded = expand_query(query, company)
            print(f"  Query expanded  : \"{expanded}\"")
            vec = embed_query(expanded, model)

            k = min(index.ntotal, per_entity * 15)
            scores, indices = index.search(vec, k)

            company_chunks = []
            for score, idx in zip(scores[0], indices[0], strict=False):
                if idx < 0 or idx >= len(metadata):
                    continue
                chunk = dict(metadata[idx])
                if chunk.get("company") != company:
                    continue
                uid = (chunk["company"], chunk.get("year"), chunk["chunk_id"])
                if uid in seen_ids:
                    continue
                seen_ids.add(uid)
                chunk["faiss_score"] = float(score)
                company_chunks.append(chunk)
                if len(company_chunks) == per_entity:
                    break

            logger.debug("Per-company: %d chunks from %s", len(company_chunks), company)
            all_candidates.extend(company_chunks)

    logger.debug("Merged pool: %d total candidates", len(all_candidates))
    return all_candidates


# ── Cohere reranking ──────────────────────────────────────────────────────────

def rerank_all(query: str, candidates: list[dict]) -> list[dict]:
    """Re-rank candidates using Cohere. Falls back to FAISS order if unavailable."""
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        logger.warning("COHERE_API_KEY not set — using FAISS order")
        return sorted(candidates, key=lambda c: c.get("faiss_score", 0), reverse=True)

    co       = cohere.Client(api_key)
    docs     = [c["text"] for c in candidates]
    response = co.rerank(
        model     = COHERE_RERANK_MODEL,
        query     = query,
        documents = docs,
        top_n     = len(docs),
    )

    scored = []
    for r in response.results:
        chunk = dict(candidates[r.index])
        chunk["rerank_score"] = r.relevance_score
        scored.append(chunk)

    return sorted(scored, key=lambda c: c["rerank_score"], reverse=True)


# ── Balanced selection ────────────────────────────────────────────────────────

def balanced_select(
    scored:    list[dict],
    top_k:     int = RERANK_TOP_K,
    min_per:   int = MIN_PER_ENTITY,
    entities:  list | None = None,
    key:       str = "company",
) -> list[dict]:
    """
    Select top_k chunks guaranteeing min_per slots for each entity.

    key: the metadata field to balance on ("company" or "year")
    entities: the specific values to guarantee slots for
    """
    if not entities:
        return scored[:top_k]

    reserved: list[dict] = []
    used_ids: set = set()

    # Reserve min_per slots per entity
    for entity in entities:
        entity_chunks = [
            c for c in scored
            if c.get(key) == entity
        ]
        for chunk in entity_chunks[:min_per]:
            uid = (chunk["company"], chunk.get("year"), chunk["chunk_id"])
            if uid not in used_ids:
                reserved.append(chunk)
                used_ids.add(uid)

    # Fill remaining slots with globally highest-scoring chunks
    remaining = top_k - len(reserved)
    for chunk in scored:
        if remaining <= 0:
            break
        uid = (chunk["company"], chunk.get("year"), chunk["chunk_id"])
        if uid not in used_ids:
            reserved.append(chunk)
            used_ids.add(uid)
            remaining -= 1

    result = sorted(reserved, key=lambda c: c.get("rerank_score", c.get("faiss_score", 0)), reverse=True)
    logger.debug(
        "Balanced selection: %d chunks from %s=%s",
        len(result), key, sorted({c.get(key) for c in result}),
    )
    return result


# ── Query classifiers ─────────────────────────────────────────────────────────

def _is_comparison_query(query: str) -> bool:
    return bool(COMPARISON_KEYWORDS.search(query))


def _is_temporal_query(query: str) -> bool:
    """Detect queries asking about evolution over time."""
    return bool(TEMPORAL_KEYWORDS.search(query))


def _detect_companies(query: str) -> list[str]:
    query_lower = query.lower()
    mentioned   = [c for c in ALL_COMPANIES if c in query_lower]
    return mentioned if mentioned else ALL_COMPANIES


def _detect_years(query: str, metadata: list[dict]) -> list[int]:
    """
    Extract year mentions from the query.
    Falls back to all available years in the index.
    """
    # Find 4-digit years in query
    mentioned = [int(y) for y in re.findall(r"\b(20\d{2})\b", query)]
    if mentioned:
        return sorted(set(mentioned))

    # "over the years" / "trend" → return all available years
    all_years = sorted({c.get("year") for c in metadata if c.get("year")})
    return all_years


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve(
    query:          str,
    index:          faiss.IndexFlatIP,
    metadata:       list[dict],
    company_filter: str | None = None,
    year_filter:    int | None = None,
) -> list[dict]:
    """
    Main entry point. Routes the query to the appropriate retrieval strategy:

      company_filter + year_filter  → exact filter (single doc)
      company_filter only           → single company, all years
      year_filter only              → single year, all companies
      temporal comparison query     → per-year balanced retrieval
      company comparison query      → per-company balanced retrieval
      standard query                → global FAISS + rerank
    """
    logger.info(
        "Retrieving: \"%s\" (company=%s, year=%s)",
        query, company_filter, year_filter,
    )

    # ── Exact filter ─────────────────────────────────────────────────────────
    if company_filter or year_filter:
        expanded = expand_query(query, company_filter)
        if company_filter:
            print(f"  Query expanded  : \"{expanded}\"")
        candidates = faiss_search(
            expanded, index, metadata,
            top_k=FAISS_TOP_K,
            company_filter=company_filter,
            year_filter=year_filter,
        )
        scored = rerank_all(query, candidates)
        return scored[:RERANK_TOP_K]

    # ── Temporal comparison query ─────────────────────────────────────────────
    if _is_temporal_query(query):
        target_years = _detect_years(query, metadata)
        # If query names a single company, restrict to that company only
        mentioned_companies = [c for c in ALL_COMPANIES if c in query.lower()]
        company_ctx = mentioned_companies[0] if len(mentioned_companies) == 1 else None
        logger.info("Temporal query → years: %s | company: %s", target_years, company_ctx or "all")

        candidates = faiss_search_per_entity(
            query, index, metadata,
            per_entity      = FAISS_PER_ENTITY,
            target_years    = target_years,
            company_context = company_ctx,
        )
        scored = rerank_all(query, candidates)
        result = balanced_select(
            scored, top_k=RERANK_TOP_K,
            min_per=MIN_PER_ENTITY,
            entities=target_years,
            key="year",
        )
        return result

    # ── Company comparison query ──────────────────────────────────────────────
    if _is_comparison_query(query):
        target_companies = _detect_companies(query)
        logger.info("Comparison query → companies: %s", target_companies)

        candidates = faiss_search_per_entity(
            query, index, metadata,
            per_entity        = FAISS_PER_ENTITY,
            target_companies  = target_companies,
        )
        scored = rerank_all(query, candidates)
        result = balanced_select(
            scored, top_k=RERANK_TOP_K,
            min_per=MIN_PER_ENTITY,
            entities=target_companies,
            key="company",
        )
        return result

    # ── Standard query ────────────────────────────────────────────────────────
    expanded = expand_query(query)
    print(f"  Query expanded  : \"{expanded}\"")
    candidates = faiss_search(expanded, index, metadata, top_k=FAISS_TOP_K)
    scored     = rerank_all(query, candidates)
    return scored[:RERANK_TOP_K]


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    index, metadata = load_index()

    # Get available years from index
    available_years = sorted({c.get("year") for c in metadata if c.get("year")})
    logger.info("Available years in index: %s", available_years)

    test_queries = [
        ("What are Tesla's main risk factors?",                    None, None),
        ("Compare Apple and Microsoft cloud revenue",              None, None),
        ("How has Tesla's AI strategy evolved over the years?",    None, None),
        ("What are Microsoft's risk factors?",                     None, 2022),
        ("Compare risk factors across all three companies",        None, None),
    ]

    for query, company_filter, year_filter in test_queries:
        logger.info("─── Query: \"%s\" ───", query)
        results = retrieve(query, index, metadata,
                           company_filter=company_filter,
                           year_filter=year_filter)

        logger.info(
            "Done — %d chunks | companies: %s | years: %s",
            len(results),
            sorted({c["company"] for c in results}),
            sorted({c.get("year") for c in results}),
        )
        for i, r in enumerate(results, 1):
            score = r.get("rerank_score", r.get("faiss_score", 0))
            logger.info(
                "  [%d] %-12s year=%-6s pages %s–%s | score=%.4f",
                i, r["company"].upper(), r.get("year", "?"),
                r["start_page"], r["end_page"], score,
            )
