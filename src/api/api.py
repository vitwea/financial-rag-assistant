"""
api.py
------
FastAPI application exposing the RAG pipeline via a REST API.

Endpoints:
    POST /ask       – answer a financial question
    GET  /health    – health check
    GET  /companies – list available companies

Run locally:
    uvicorn src.api.api:app --reload --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.pipeline.pipeline import RAGPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Lifespan ──────────────────────────────────────────────────────────────────

pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app):
    global pipeline
    logger.info("Starting up — loading RAG pipeline...")
    pipeline = RAGPipeline()
    logger.info("API ready")
    yield


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Financial RAG Assistant",
    lifespan=lifespan,
    description=(
        "Ask complex questions about Tesla, Apple, and Microsoft "
        "annual reports (10-K filings). Every answer is grounded in "
        "real document passages with page-level citations."
    ),
    version="1.0.0",
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str = Field(
        ..., min_length=5, max_length=500,
        json_schema_extra={"example": "What are the main risk factors for Tesla?"},
    )
    company_filter: str | None = Field(
        default=None,
        description="Restrict to one company: 'tesla', 'apple', or 'microsoft'",
        json_schema_extra={"example": "tesla"},
    )
    year_filter: int | None = Field(
        default=None,
        description="Restrict to a specific fiscal year, e.g. 2022",
        json_schema_extra={"example": 2022},
    )


class SourceRecord(BaseModel):
    company:    str
    source:     str
    start_page: int
    end_page:   int
    score:      float


class AskResponse(BaseModel):
    query:          str
    answer:         str
    sources:        str
    source_details: list[SourceRecord]
    latency_ms:     int


class HealthResponse(BaseModel):
    status:       str
    index_loaded: bool
    total_chunks: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Returns API status and index statistics."""
    if pipeline is None:
        return HealthResponse(status="starting", index_loaded=False, total_chunks=0)
    return HealthResponse(
        status="ok",
        index_loaded=True,
        total_chunks=len(pipeline.metadata),
    )


@app.get("/companies", tags=["System"])
def list_companies():
    """Returns the list of companies available for querying."""
    return {"companies": ["tesla", "apple", "microsoft"]}


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
def ask(request: AskRequest):
    """Answer a financial question using the RAG pipeline."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised yet.")

    valid_companies = {"tesla", "apple", "microsoft"}
    if request.company_filter and request.company_filter not in valid_companies:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid company_filter. Choose from: {sorted(valid_companies)}",
        )

    logger.info("POST /ask — query: \"%s\" | filter: %s",
                request.query, request.company_filter)

    start  = time.perf_counter()
    result = pipeline.ask(request.query, company_filter=request.company_filter, year_filter=request.year_filter)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    logger.info("Response generated in %d ms", elapsed_ms)

    source_details = [
        SourceRecord(
            company    = chunk["company"],
            source     = chunk["source"],
            start_page = chunk["start_page"],
            end_page   = chunk["end_page"],
            score      = round(chunk.get("rerank_score",
                                         chunk.get("faiss_score", 0.0)), 4),
        )
        for chunk in result["chunks_used"]
    ]

    return AskResponse(
        query          = result["query"],
        answer         = result["answer"],
        sources        = result["sources"],
        source_details = source_details,
        latency_ms     = elapsed_ms,
    )
