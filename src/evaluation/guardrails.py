"""
guardrails.py
-------------
Pre- and post-generation safety checks for the RAG pipeline.

Pre-generation guardrails (run BEFORE calling the LLM):
  - Off-topic query  : question is not about Tesla, Apple, or Microsoft.
  - No chunks        : retriever returned nothing.
  - Low confidence   : best chunk score is below the minimum threshold.

Post-generation guardrails (run AFTER the LLM responds):
  - Hallucination signal : LLM response contains phrases that indicate
                           it went beyond the provided context.
  - Citation check       : answer must include at least one page citation.

Usage:
    from src.evaluation.guardrails import PreGuardrail, PostGuardrail
"""

import re

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

# Minimum rerank score — below this the retrieved context is too weak
MIN_RERANK_SCORE = 0.01   # raised from 0.001 (was effectively never triggered)

# Minimum FAISS score when Cohere reranker is not available
MIN_FAISS_SCORE  = 0.15

SUPPORTED_COMPANIES = {"tesla", "apple", "microsoft"}

FINANCIAL_KEYWORDS = re.compile(
    r"\b(tesla|apple|microsoft|10-k|annual report|revenue|profit|margin|"
    r"risk|earnings|cloud|azure|iphone|ev|electric vehicle|stock|"
    r"segment|guidance|eps|ebitda|cash flow|balance sheet|income|"
    r"operating|growth|services|hardware|software|ai|datacenter)\b",
    re.IGNORECASE,
)

INVESTMENT_ADVICE_PATTERNS = re.compile(
    r"\b(should i (buy|sell|invest|short|hold)|"
    r"is it (worth|safe|good) to (buy|invest|own)|"
    r"(buy|sell|invest in|short|hold) (tesla|apple|microsoft|tsla|aapl|msft)|"
    r"(good|bad|worth) investment|"
    r"investment advice|financial advice|"
    r"will (the )?stock|price target|"
    r"(going to|will it) (go up|go down|rise|fall|crash))\b",
    re.IGNORECASE,
)

HALLUCINATION_PHRASES = [
    "based on my knowledge",
    "i believe",
    "i think",
    "as of my last update",
    "generally speaking",
    "it is commonly known",
    "typically companies",
    "in my experience",
    "from what i know",
]


# ── Result ────────────────────────────────────────────────────────────────────

class GuardrailResult:
    """
    Outcome of a single guardrail check.

    Attributes:
        passed   : True if the check passed (pipeline can continue)
        reason   : machine-readable label when the check fails
        message  : user-facing message returned when failed
    """

    def __init__(self, passed: bool, reason: str = "", message: str = ""):
        self.passed  = passed
        self.reason  = reason
        self.message = message

    def __bool__(self) -> bool:
        return self.passed

    def __repr__(self) -> str:
        status = "PASS" if self.passed else f"FAIL({self.reason})"
        return f"GuardrailResult({status})"


# ── Pre-generation guardrails ─────────────────────────────────────────────────

class PreGuardrail:
    """Checks to run before calling the LLM (fast, no API calls)."""

    @staticmethod
    def check_has_chunks(chunks: list[dict]) -> GuardrailResult:
        if not chunks:
            logger.warning("Guardrail FAIL: no chunks retrieved")
            return GuardrailResult(
                passed  = False,
                reason  = "no_chunks",
                message = (
                    "I could not find any relevant passages in the 10-K filings "
                    "to answer your question. Please try rephrasing your query or "
                    "ask about Tesla, Apple, or Microsoft specifically."
                ),
            )
        return GuardrailResult(passed=True)

    @staticmethod
    def check_confidence(chunks: list[dict]) -> GuardrailResult:
        """Block if the best chunk score is below the minimum threshold."""
        best_score = max(
            c.get("rerank_score", c.get("rrf_score", c.get("faiss_score", 0.0)))
            for c in chunks
        )

        threshold = (
            MIN_RERANK_SCORE
            if any("rerank_score" in c for c in chunks)
            else MIN_FAISS_SCORE
        )

        if best_score < threshold:
            logger.warning(
                "Guardrail FAIL: low confidence (best_score=%.4f < threshold=%.4f)",
                best_score, threshold,
            )
            return GuardrailResult(
                passed  = False,
                reason  = f"low_confidence(score={best_score:.4f})",
                message = (
                    "The retrieved passages have very low relevance scores for your "
                    "question, which means the 10-K documents may not contain the "
                    "information you are looking for. "
                    "Please try a more specific question about Tesla, Apple, or Microsoft."
                ),
            )

        logger.info("Guardrail PASS: confidence OK (best_score=%.4f)", best_score)
        return GuardrailResult(passed=True)

    @staticmethod
    def check_on_topic(query: str) -> GuardrailResult:
        """Reject off-topic queries and investment advice requests."""
        query_lower = query.lower()

        if bool(INVESTMENT_ADVICE_PATTERNS.search(query)):
            logger.warning("Guardrail FAIL: investment advice request: '%s'", query)
            return GuardrailResult(
                passed  = False,
                reason  = "investment_advice",
                message = (
                    "This assistant analyses 10-K filings and cannot provide "
                    "investment advice or stock recommendations. "
                    "Please ask about specific financial metrics, strategies, or "
                    "risks disclosed in Tesla, Apple, or Microsoft annual reports."
                ),
            )

        mentions_company = any(c in query_lower for c in SUPPORTED_COMPANIES)
        mentions_finance = bool(FINANCIAL_KEYWORDS.search(query))

        if not mentions_company and not mentions_finance:
            logger.warning("Guardrail FAIL: off-topic query: '%s'", query)
            return GuardrailResult(
                passed  = False,
                reason  = "off_topic",
                message = (
                    "This assistant specialises in the annual reports (10-K filings) "
                    "of Tesla, Apple, and Microsoft. "
                    "Please ask a question related to these companies or their financials."
                ),
            )

        return GuardrailResult(passed=True)

    @classmethod
    def run_all(cls, query: str, chunks: list[dict]) -> GuardrailResult:
        """Run all pre-generation checks. Short-circuits on first failure."""
        for check in [
            cls.check_on_topic(query),
            cls.check_has_chunks(chunks),
        ]:
            if not check:
                return check

        confidence = cls.check_confidence(chunks)
        if not confidence:
            return confidence

        logger.info("All pre-generation guardrails passed")
        return GuardrailResult(passed=True)


# ── Post-generation guardrails ────────────────────────────────────────────────

class PostGuardrail:
    """Checks to run after the LLM generates a response."""

    @staticmethod
    def check_no_hallucination_phrases(answer: str) -> GuardrailResult:
        """Flag responses that contain typical out-of-context LLM phrases."""
        answer_lower = answer.lower()
        found = [p for p in HALLUCINATION_PHRASES if p in answer_lower]

        if found:
            logger.warning("Guardrail WARN: possible hallucination phrases: %s", found)
            return GuardrailResult(
                passed  = False,
                reason  = f"hallucination_phrases({found})",
                message = (
                    "⚠️  Note: The answer may contain information not grounded in "
                    "the provided 10-K documents. Please verify against the source pages."
                ),
            )

        return GuardrailResult(passed=True)

    @staticmethod
    def check_has_citations(answer: str) -> GuardrailResult:
        """Warn if the answer contains no inline [Company | pages X–Y] citations."""
        citation_pattern = re.compile(
            r"\[(Tesla|Apple|Microsoft)\s*\|\s*pages?\s*\d+",
            re.IGNORECASE,
        )
        if not citation_pattern.search(answer):
            logger.warning("Guardrail WARN: no citations found in answer")
            return GuardrailResult(
                passed  = False,
                reason  = "no_citations",
                message = (
                    "⚠️  Note: This answer does not include source citations. "
                    "Treat it with caution."
                ),
            )

        return GuardrailResult(passed=True)

    @classmethod
    def run_all(cls, answer: str) -> list[GuardrailResult]:
        """Run all post-generation checks. Collects all warnings (no short-circuit)."""
        results = [
            cls.check_no_hallucination_phrases(answer),
            cls.check_has_citations(answer),
        ]

        failures = [r for r in results if not r]
        if failures:
            logger.warning(
                "Post-generation guardrail warnings: %s",
                [r.reason for r in failures],
            )
        else:
            logger.info("All post-generation guardrails passed")

        return results