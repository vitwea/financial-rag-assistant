"""
guardrails.py
-------------
Pre- and post-generation safety checks for the RAG pipeline.

Pre-generation guardrails (run BEFORE calling the LLM):
  - Low confidence  : all retrieved chunks have very low rerank scores,
                      meaning the index likely has no relevant content.
  - Off-topic query : question is not about Tesla, Apple, or Microsoft.
  - No chunks       : retriever returned nothing.

Post-generation guardrails (run AFTER the LLM responds):
  - Hallucination signal : LLM response contains phrases that indicate
                           it went beyond the provided context.

If any pre-generation check fails, the pipeline short-circuits and returns
a safe refusal message instead of calling the expensive LLM.

Usage:
    from src.evaluation.guardrails import PreGuardrail, PostGuardrail
"""

import re

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

# If the best rerank score is below this threshold, context is too weak
MIN_RERANK_SCORE  = 0.001   # comparison queries produce lower scores by design

# If the best FAISS score is below this threshold (when no reranker is used)
MIN_FAISS_SCORE   = 0.15

# Companies the assistant is allowed to answer questions about
SUPPORTED_COMPANIES = {"tesla", "apple", "microsoft"}

# Keywords that suggest the query is about our supported companies or
# general financial topics (used to detect off-topic questions)
FINANCIAL_KEYWORDS = re.compile(
    r"\b(tesla|apple|microsoft|10-k|annual report|revenue|profit|margin|"
    r"risk|earnings|cloud|azure|iphone|ev|electric vehicle|stock|"
    r"segment|guidance|eps|ebitda|cash flow|balance sheet|income|"
    r"operating|growth|services|hardware|software|ai|datacenter)\b",
    re.IGNORECASE,
)

# Patterns that indicate investment advice requests — always blocked
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

# Phrases that indicate the LLM went beyond the provided context
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


# ── Result dataclass ──────────────────────────────────────────────────────────

class GuardrailResult:
    """
    Outcome of a guardrail check.

    Attributes:
        passed   : True if the check passed (pipeline can continue)
        reason   : human-readable explanation when the check fails
        message  : safe response to return to the user when failed
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
    """
    Checks to run before calling the LLM.
    All checks are fast and cheap (no API calls).
    """

    @staticmethod
    def check_has_chunks(chunks: list[dict]) -> GuardrailResult:
        """Fail if retriever returned no chunks at all."""
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
        """
        Fail if the best chunk score is below the minimum threshold.
        This prevents the LLM from hallucinating when no relevant context exists.
        """
        best_score = max(
            c.get("rerank_score", c.get("faiss_score", 0.0))
            for c in chunks
        )

        # Use the appropriate threshold depending on which scorer was used
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
        """
        Fail if the query does not appear to be about supported companies
        or financial topics. Prevents using the assistant as a general chatbot.
        Also blocks investment advice requests.
        """
        query_lower = query.lower()

        # Block investment advice requests regardless of company mention
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

        # Check for supported company names
        mentions_company = any(c in query_lower for c in SUPPORTED_COMPANIES)

        # Check for financial keywords
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
        """
        Run all pre-generation checks in order.
        Returns the first failure encountered, or a passing result.
        """
        # Run checks sequentially — short-circuit on first failure
        for check in [
            cls.check_on_topic(query),
            cls.check_has_chunks(chunks),
        ]:
            if not check:
                return check

        # Only run confidence check if we actually have chunks
        confidence = cls.check_confidence(chunks)
        if not confidence:
            return confidence

        logger.info("All pre-generation guardrails passed")
        return GuardrailResult(passed=True)


# ── Post-generation guardrails ────────────────────────────────────────────────

class PostGuardrail:
    """
    Checks to run after the LLM generates a response.
    Detects answers that went beyond the provided context.
    """

    @staticmethod
    def check_no_hallucination_phrases(answer: str) -> GuardrailResult:
        """
        Flag responses that contain phrases typically used when an LLM
        is drawing on training knowledge rather than the provided context.
        """
        answer_lower = answer.lower()
        found = [p for p in HALLUCINATION_PHRASES if p in answer_lower]

        if found:
            logger.warning(
                "Guardrail WARN: possible hallucination phrases detected: %s", found
            )
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
        """
        Warn if the answer contains no inline citations in the expected format.
        A well-grounded answer should always cite [Company | pages X-Y].
        """
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
        """
        Run all post-generation checks.
        Returns a list of all results (both passes and failures).
        Unlike pre-generation, we don't short-circuit — we collect all warnings.
        """
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
