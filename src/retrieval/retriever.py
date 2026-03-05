"""
retriever.py
------------
Three-stage hybrid retrieval pipeline.

  Stage 1 — Hybrid search (FAISS + BM25 → Reciprocal Rank Fusion)
      FAISS handles semantic similarity.
      BM25 handles exact keywords, numbers, and named entities.
      RRF merges both ranked lists into a single candidate pool.

  Stage 2 — Re-ranking (Cohere)
      Cross-encoder re-scores merged candidates against the query.

  Stage 3 — Balanced selection (comparison queries only)
      Guarantees representation from all requested companies/years.

Query routing:
  - company_filter / year_filter set  → exact filter
  - Temporal comparison query         → per-year balanced retrieval
  - Company comparison query          → per-company balanced retrieval
  - Standard query                    → global hybrid search
"""

import os
import pickle
import re
from pathlib import Path

import cohere
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

from src.retrieval.bm25_index import build_bm25_index, bm25_search
from src.retrieval.query_expander import expand_query
from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

INDEX_DIR  = Path("data/index")
MODEL_NAME = "text-embedding-3-small"

FAISS_TOP_K      = 20
BM25_TOP_K       = 20
RRF_K            = 60
RERANK_TOP_K     = 5
FAISS_PER_ENTITY = 8
MIN_PER_ENTITY   = 1

COHERE_RERANK_MODEL = "rerank-english-v3.0"

ALL_COMPANIES = ["tesla", "apple", "microsoft"]

COMPARISON_KEYWORDS = re.compile(
    r"\b(compare|comparison|versus|vs\.?|between|all (three|companies)|each company)\b",
    re.IGNORECASE,
)

TEMPORAL_KEYWORDS = re.compile(
    r"\b(evolution|evolve|evolved|trend|over the years|over time|"
    r"year.over.year|yoy|historically|history|changed|change|"
    r"progress|progression|compared to \d{4}|\d{4}.+(vs|versus|compared)|\d{4}.+\d{4})\b",
    re.IGNORECASE,
)


# ── OpenAI client (lazy singleton) ────────────────────────────────────────────

_openai_client: OpenAI | None = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise OSError("OPENAI_API_KEY not set in .env")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


# ── Index loading ─────────────────────────────────────────────────────────────

def load_index() -> tuple[faiss.IndexFlatIP, list[dict], BM25Okapi]:
    """
    Load FAISS index + metadata and build BM25 index in memory.
    Returns (faiss_index, metadata, bm25_index).
    """
    index_path    = INDEX_DIR / "faiss.index"
    metadata_path = INDEX_DIR / "metadata.pkl"

    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Index not found. Run src/embeddings/embeddings.py first."
        )

    index = faiss.read_index(str(index_path))
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    bm25 = build_bm25_index(metadata)

    logger.info(
        "FAISS index: %d vectors | BM25 index: %d chunks",
        index.ntotal, len(metadata),
    )
    return index, metadata, bm25


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_query(query: str) -> np.ndarray:
    """Embed a single query via OpenAI. Returns normalised (1, 1536) float32."""
    client   = _get_openai_client()
    response = client.embeddings.create(model=MODEL_NAME, input=query)
    vec      = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec


# ── FAISS search ──────────────────────────────────────────────────────────────

def faiss_search(
    query:          str,
    index:          faiss.IndexFlatIP,
    metadata:       list[dict],
    top_k:          int = FAISS_TOP_K,
    company_filter: str | None = None,
    year_filter:    int | None = None,
) -> list[dict]:
    """Semantic vector search with optional filters."""
    vec = embed_query(query)
    k   = min(
        index.ntotal,
        top_k * 10 if (company_filter or year_filter) else top_k,
    )
    scores, indices = index.search(vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0], strict=False):
        if idx < 0 or idx >= len(metadata):
            continue
        chunk = dict(metadata[idx])
        if company_filter and chunk.get("company") != company_filter:
            continue
        if year_filter and chunk.get("year") != year_filter:
            continue
        chunk["faiss_score"] = float(score)
        results.append(chunk)
        if len(results) == top_k:
            break

    logger.debug(
        "FAISS: %d candidates (company=%s year=%s)",
        len(results), company_filter, year_filter,
    )
    return results


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    faiss_results: list[dict],
    bm25_results:  list[dict],
    k:             int = RRF_K,
) -> list[dict]:
    """
    Merge FAISS and BM25 ranked lists using Reciprocal Rank Fusion.

    RRF score = Σ 1 / (k + rank_i)

    Sorting is stable: ties are broken by uid string so the output
    is always deterministic for the same inputs.
    """
    def _uid(chunk: dict) -> tuple:
        return (chunk["company"], chunk.get("year"), chunk["chunk_id"])

    rrf_scores: dict[tuple, float] = {}
    chunks_map: dict[tuple, dict]  = {}

    for rank, chunk in enumerate(faiss_results, start=1):
        uid = _uid(chunk)
        rrf_scores[uid] = rrf_scores.get(uid, 0.0) + 1.0 / (k + rank)
        chunks_map[uid] = chunk

    for rank, chunk in enumerate(bm25_results, start=1):
        uid = _uid(chunk)
        rrf_scores[uid] = rrf_scores.get(uid, 0.0) + 1.0 / (k + rank)
        if uid not in chunks_map:
            chunks_map[uid] = chunk

    # Secondary sort by str(uid) breaks ties deterministically
    merged = []
    for uid, score in sorted(
        rrf_scores.items(),
        key=lambda x: (x[1], str(x[0])),
        reverse=True,
    ):
        chunk = dict(chunks_map[uid])
        chunk["rrf_score"] = round(score, 6)
        merged.append(chunk)

    logger.debug(
        "RRF: %d FAISS + %d BM25 → %d unique candidates",
        len(faiss_results), len(bm25_results), len(merged),
    )
    return merged


# ── Hybrid search ─────────────────────────────────────────────────────────────

def hybrid_search(
    query:          str,
    index:          faiss.IndexFlatIP,
    metadata:       list[dict],
    bm25:           BM25Okapi,
    top_k:          int = FAISS_TOP_K,
    company_filter: str | None = None,
    year_filter:    int | None = None,
) -> list[dict]:
    """FAISS + BM25 merged via RRF."""
    faiss_results = faiss_search(
        query, index, metadata, top_k=top_k,
        company_filter=company_filter, year_filter=year_filter,
    )
    bm25_results = bm25_search(
        query, bm25, metadata, top_k=top_k,
        company_filter=company_filter, year_filter=year_filter,
    )
    return reciprocal_rank_fusion(faiss_results, bm25_results)


# ── Per-entity hybrid search ──────────────────────────────────────────────────

def hybrid_search_per_entity(
    query:            str,
    index:            faiss.IndexFlatIP,
    metadata:         list[dict],
    bm25:             BM25Okapi,
    per_entity:       int = FAISS_PER_ENTITY,
    target_companies: list[str] | None = None,
    target_years:     list[int] | None = None,
    company_context:  str | None = None,
) -> list[dict]:
    """Hybrid search per company or per year to guarantee balanced pools."""
    all_candidates: list[dict] = []
    seen_ids: set[tuple]       = set()

    def _uid(chunk: dict) -> tuple:
        return (chunk["company"], chunk.get("year"), chunk["chunk_id"])

    targets   = target_years if target_years else (target_companies or [])
    use_years = bool(target_years)

    for entity in targets:
        cf = company_context if use_years else entity
        yf = entity if use_years else None

        expanded = expand_query(query, cf)
        results  = hybrid_search(
            expanded, index, metadata, bm25,
            top_k=per_entity * 3,
            company_filter=cf,
            year_filter=yf,
        )

        added = 0
        for chunk in results:
            uid = _uid(chunk)
            if uid in seen_ids:
                continue
            seen_ids.add(uid)
            all_candidates.append(chunk)
            added += 1
            if added == per_entity:
                break

    logger.debug("Merged pool: %d candidates", len(all_candidates))
    return all_candidates


# ── Cohere reranking ──────────────────────────────────────────────────────────

def rerank_all(query: str, candidates: list[dict]) -> list[dict]:
    """Re-rank via Cohere. Falls back to RRF order if key not set."""
    if not candidates:
        return []

    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        logger.warning("COHERE_API_KEY not set — using RRF order")
        return sorted(
            candidates,
            key=lambda c: (c.get("rrf_score", c.get("faiss_score", 0)), str(c.get("chunk_id", ""))),
            reverse=True,
        )

    co       = cohere.Client(api_key)
    response = co.rerank(
        model     = COHERE_RERANK_MODEL,
        query     = query,
        documents = [c["text"] for c in candidates],
        top_n     = len(candidates),
    )
    scored = []
    for r in response.results:
        chunk = dict(candidates[r.index])
        chunk["rerank_score"] = r.relevance_score
        scored.append(chunk)

    # Stable sort: ties broken by chunk_id
    return sorted(
        scored,
        key=lambda c: (c["rerank_score"], str(c.get("chunk_id", ""))),
        reverse=True,
    )


# ── Balanced selection ────────────────────────────────────────────────────────

def balanced_select(
    scored:   list[dict],
    top_k:    int = RERANK_TOP_K,
    min_per:  int = MIN_PER_ENTITY,
    entities: list | None = None,
    key:      str = "company",
) -> list[dict]:
    """Select top_k guaranteeing min_per slots per entity."""
    if not entities:
        return scored[:top_k]

    reserved: list[dict] = []
    used_ids: set        = set()

    def _uid(c: dict) -> tuple:
        return (c["company"], c.get("year"), c["chunk_id"])

    for entity in entities:
        for chunk in [c for c in scored if c.get(key) == entity][:min_per]:
            uid = _uid(chunk)
            if uid not in used_ids:
                reserved.append(chunk)
                used_ids.add(uid)

    for chunk in scored:
        if len(reserved) >= top_k:
            break
        uid = _uid(chunk)
        if uid not in used_ids:
            reserved.append(chunk)
            used_ids.add(uid)

    return sorted(
        reserved,
        key=lambda c: (c.get("rerank_score", c.get("rrf_score", 0)), str(c.get("chunk_id", ""))),
        reverse=True,
    )


# ── Query classifiers ─────────────────────────────────────────────────────────

def _is_comparison_query(query: str) -> bool:
    return bool(COMPARISON_KEYWORDS.search(query))


def _is_temporal_query(query: str) -> bool:
    return bool(TEMPORAL_KEYWORDS.search(query))


def _detect_companies(query: str) -> list[str]:
    """
    Detect companies explicitly mentioned in the query.
    Returns [] if none are mentioned — no fallback to ALL_COMPANIES.
    """
    low   = query.lower()
    found = [c for c in ALL_COMPANIES if c in low]
    return found


def _detect_years(query: str, metadata: list[dict]) -> list[int]:
    mentioned = [int(y) for y in re.findall(r"\b(20\d{2})\b", query)]
    if mentioned:
        return sorted(set(mentioned))
    return sorted({c.get("year") for c in metadata if c.get("year")})


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve(
    query:          str,
    index:          faiss.IndexFlatIP,
    metadata:       list[dict],
    bm25:           BM25Okapi,
    company_filter: str | None = None,
    year_filter:    int | None = None,
) -> list[dict]:
    """
    Main entry point — routes the query and returns top-k chunks for the LLM.
    """
    logger.info(
        'Retrieving: "%s" (company=%s year=%s)',
        query, company_filter, year_filter,
    )

    # 1) Explicit filter by company/year → always respect
    if company_filter or year_filter:
        expanded   = expand_query(query, company_filter)
        candidates = hybrid_search(
            expanded, index, metadata, bm25,
            top_k=FAISS_TOP_K,
            company_filter=company_filter,
            year_filter=year_filter,
        )
        return rerank_all(query, candidates)[:RERANK_TOP_K]

    # 2) Temporal queries (trends, historical evolution, etc.)
    if _is_temporal_query(query):
        companies   = _detect_companies(query)
        company_ctx = companies[0] if len(companies) == 1 else None
        years       = _detect_years(query, metadata)
        candidates  = hybrid_search_per_entity(
            query, index, metadata, bm25,
            target_years=years, company_context=company_ctx,
        )
        scored = rerank_all(query, candidates)
        return balanced_select(scored, entities=years, key="year")

    # 3) Comparison queries
    if _is_comparison_query(query):
        companies = _detect_companies(query)

        # "all three companies" / "each company" / no names → use all three
        if not companies:
            companies = ALL_COMPANIES

        candidates = hybrid_search_per_entity(
            query, index, metadata, bm25,
            target_companies=companies,
        )
        scored = rerank_all(query, candidates)
        return balanced_select(scored, entities=companies, key="company")

    # 4) Standard query
    candidates = hybrid_search(query, index, metadata, bm25, top_k=FAISS_TOP_K)
    return rerank_all(query, candidates)[:RERANK_TOP_K]