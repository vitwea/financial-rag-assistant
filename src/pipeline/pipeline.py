"""
pipeline.py
-----------
Orchestrates the full RAG pipeline:

    User query
        ↓
    Pre-guardrails    (topic check, confidence check)
        ↓
    Retriever         (Hybrid FAISS+BM25 → RRF → Cohere rerank)
        ↓
    LLM generation    (GPT-4o-mini with grounded prompt)
        ↓
    Post-guardrails   (hallucination detection, citation check)
        ↓
    Evaluator         (Claude judge — optional, adds latency)
        ↓
    Response          (answer + sources + quality scores)

Usage:
    python -m src.pipeline.pipeline
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

from src.evaluation.guardrails import PostGuardrail, PreGuardrail
from src.retrieval.retriever import load_index, retrieve
from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

LLM_MODEL   = "gpt-4o-mini"
MAX_TOKENS  = 1_024
TEMPERATURE = 0.1

SYSTEM_PROMPT = """You are a professional financial analyst assistant specializing
in annual reports (10-K filings) for Tesla, Apple, and Microsoft.

Rules you must ALWAYS follow:
1. Base every statement EXCLUSIVELY on the context passages provided below.
2. After each factual claim, cite the source in the format [Company | pages X–Y].
3. If the context does not contain enough information to answer, reply:
   "The provided documents do not contain sufficient information to answer this question."
4. Never invent numbers, dates, or facts not present in the context.
5. When comparing companies, structure your answer with clear headings per company.
6. Be concise but thorough. Use bullet points for lists of risks or metrics."""


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(query: str, chunks: list[dict]) -> str:
    """Construct the user message with retrieved context injected."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        header = (
            f"--- Passage {i} "
            f"({chunk['company'].capitalize()} | "
            f"pages {chunk['start_page']}–{chunk['end_page']}) ---"
        )
        context_parts.append(f"{header}\n{chunk['text']}")

    context_block = "\n\n".join(context_parts)
    return f"[Context]\n{context_block}\n\n[Question]\n{query}"


# ── LLM call ──────────────────────────────────────────────────────────────────

def call_llm(system: str, user: str, client: OpenAI) -> str:
    """Send a chat completion request and return the response text."""
    logger.info("Calling LLM: %s", LLM_MODEL)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return response.choices[0].message.content


# ── Source formatter ──────────────────────────────────────────────────────────

def format_sources(chunks: list[dict]) -> str:
    """Build a deduplicated source list appended to every answer."""
    seen  = set()
    lines = ["**Sources used:**"]

    for chunk in chunks:
        key = (chunk["company"], chunk["start_page"], chunk["end_page"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"  • {chunk['company'].capitalize()} 10-K | "
            f"pages {chunk['start_page']}–{chunk['end_page']} | "
            f"{chunk['source']}"
        )

    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    High-level interface for the RAG assistant with guardrails and
    optional automatic evaluation.

    Example:
        pipeline = RAGPipeline()
        result = pipeline.ask("What are Tesla's main risk factors?")
        result = pipeline.ask("What are Tesla's main risk factors?", evaluate=True)
    """

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise OSError("OPENAI_API_KEY not set in .env")

        self.client = OpenAI(api_key=api_key)
        # load_index now returns (faiss_index, metadata, bm25_index)
        self.index, self.metadata, self.bm25 = load_index()
        self._evaluator = None
        logger.info("RAGPipeline ready (hybrid FAISS+BM25 retrieval)")

    def _get_evaluator(self):
        if self._evaluator is None:
            from src.evaluation.evaluator import RAGEvaluator
            self._evaluator = RAGEvaluator()
        return self._evaluator

    def ask(
        self,
        query:          str,
        company_filter: str | None = None,
        year_filter:    int | None = None,
        evaluate:       bool = False,
    ) -> dict:
        """
        Answer a financial question using the RAG pipeline.

        Args:
            query          : natural-language question
            company_filter : restrict retrieval to one company
            year_filter    : restrict retrieval to one fiscal year
            evaluate       : run Claude judge after generation (adds ~2s latency)

        Returns dict with keys:
            query, answer, sources, chunks_used, guardrails, evaluation, blocked
        """
        logger.info("Pipeline.ask: \"%s\" (company=%s, year=%s, evaluate=%s)",
                    query, company_filter, year_filter, evaluate)

        # ── Step 1: Pre-guardrail — topic check ───────────────────────────────
        topic_check = PreGuardrail.check_on_topic(query)
        if not topic_check:
            logger.warning("Blocked by topic guardrail: %s", topic_check.reason)
            return self._blocked_response(query, topic_check.message, [topic_check])

        # ── Step 2: Retrieve chunks (hybrid FAISS + BM25 + Cohere rerank) ─────
        chunks = retrieve(
            query, self.index, self.metadata, self.bm25,
            company_filter=company_filter,
            year_filter=year_filter,
        )

        # ── Step 3: Pre-guardrail — context quality check ─────────────────────
        context_check = PreGuardrail.run_all(query, chunks)
        if not context_check:
            logger.warning("Blocked by context guardrail: %s", context_check.reason)
            return self._blocked_response(query, context_check.message, [context_check])

        # ── Step 4: Generate answer ───────────────────────────────────────────
        user_prompt = build_prompt(query, chunks)
        answer      = call_llm(SYSTEM_PROMPT, user_prompt, self.client)
        sources     = format_sources(chunks)
        logger.info("Answer generated (%d chars)", len(answer))

        # ── Step 5: Post-guardrails ───────────────────────────────────────────
        post_results  = PostGuardrail.run_all(answer)
        post_warnings = [r.message for r in post_results if not r]
        if post_warnings:
            answer += "\n\n" + "\n".join(post_warnings)

        # ── Step 6: Optional evaluation ───────────────────────────────────────
        evaluation = None
        if evaluate:
            try:
                evaluation = self._get_evaluator().evaluate(query, answer, chunks)
            except Exception as exc:
                logger.error("Evaluation failed: %s", exc)

        return {
            "query":       query,
            "answer":      answer,
            "sources":     sources,
            "chunks_used": chunks,
            "guardrails":  post_results,
            "evaluation":  evaluation,
            "blocked":     False,
        }

    @staticmethod
    def _blocked_response(query: str, message: str, guardrails: list) -> dict:
        return {
            "query":       query,
            "answer":      message,
            "sources":     "",
            "chunks_used": [],
            "guardrails":  guardrails,
            "evaluation":  None,
            "blocked":     True,
        }