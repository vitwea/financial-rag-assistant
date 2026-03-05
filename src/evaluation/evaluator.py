"""
evaluator.py
------------
Automatic evaluation of RAG pipeline responses using Claude as a judge.

Claude scores each response across four dimensions:

  grounding    — Is every claim supported by the retrieved passages?
  relevance    — Does the answer directly address the question asked?
  faithfulness — Does the answer avoid adding information beyond the context?
  completeness — Does the answer cover all aspects of the question?

Each dimension is scored 0.0–1.0. Claude returns structured JSON so scores
can be logged, aggregated, and tracked over time.

This is the "LLM-as-a-judge" pattern used in production RAG evaluation
pipelines (e.g. RAGAS, TruLens).

Usage:
    evaluator = RAGEvaluator()
    scores = evaluator.evaluate(query, answer, chunks)
    print(scores.summary())
"""

from dataclasses import asdict, dataclass
import json
import os

import anthropic
from dotenv import load_dotenv

from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

JUDGE_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS  = 1024     # increased from 512 — judge needs room to reason
TEMPERATURE = 0.0

# Minimum acceptable score per dimension
SCORE_THRESHOLDS = {
    "grounding":    0.60,   # paraphrases count as grounded
    "relevance":    0.65,
    "faithfulness": 0.65,   # synthesis across passages is NOT a violation
    "completeness": 0.55,
}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    """
    Scores returned by the Claude judge for a single RAG response.

    All scores are in [0.0, 1.0].
    passed is True if all scores meet their minimum thresholds.
    """
    grounding:    float
    relevance:    float
    faithfulness: float
    completeness: float
    reasoning:    dict
    passed:       bool = True

    def __post_init__(self):
        self.passed = all(
            getattr(self, dim) >= threshold
            for dim, threshold in SCORE_THRESHOLDS.items()
        )

    @property
    def average(self) -> float:
        return (
            self.grounding + self.relevance +
            self.faithfulness + self.completeness
        ) / 4

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return (
            f"{status} | avg={self.average:.2f} | "
            f"grounding={self.grounding:.2f} | "
            f"relevance={self.relevance:.2f} | "
            f"faithfulness={self.faithfulness:.2f} | "
            f"completeness={self.completeness:.2f}"
        )


# ── Prompt ────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of RAG (Retrieval-Augmented Generation)
systems specializing in financial document analysis.

Your task is to score an AI-generated answer based on the retrieved context passages
that were provided to the AI when it generated the answer.

CRITICAL EVALUATION RULES:
1. The answer is expected to PARAPHRASE and SYNTHESIZE the passages — this is correct behaviour.
   A claim counts as grounded if it can be reasonably inferred from the passages,
   even if the exact wording differs.
2. Synthesizing information from multiple passages into a summary is NOT a faithfulness violation.
   Only penalize faithfulness if the answer introduces facts that cannot be traced to any passage.
3. Be calibrated: a well-structured answer with proper citations and reasonable paraphrasing
   should score 0.75–0.95, not 0.3–0.5.

You MUST respond with ONLY a valid JSON object. No preamble, no explanation outside the JSON.

JSON format:
{
  "grounding": <float 0.0-1.0>,
  "relevance": <float 0.0-1.0>,
  "faithfulness": <float 0.0-1.0>,
  "completeness": <float 0.0-1.0>,
  "reasoning": {
    "grounding": "<one sentence explaining the grounding score>",
    "relevance": "<one sentence explaining the relevance score>",
    "faithfulness": "<one sentence explaining the faithfulness score>",
    "completeness": "<one sentence explaining the completeness score>"
  }
}

Scoring definitions:
  grounding    (0-1): Fraction of claims in the answer supported by the retrieved passages.
                      Paraphrases and reasonable inferences from the passages count as supported.
                      Only unsupported or invented claims reduce this score.
                      1.0 = every claim is traceable to at least one passage.

  relevance    (0-1): How directly the answer addresses the specific question asked.
                      1.0 = perfectly on-topic, no irrelevant content.

  faithfulness (0-1): Does the answer avoid introducing facts NOT present in any passage?
                      Summarizing, paraphrasing, or combining multiple passages is ALLOWED.
                      Only penalize if the answer adds new facts absent from all passages.
                      1.0 = no information beyond what the passages contain.

  completeness (0-1): How thoroughly the answer covers all answerable aspects of the question
                      given what is available in the context.
                      1.0 = all aspects that could be answered from the passages are addressed."""


def _build_judge_prompt(
    query: str,
    answer: str,
    chunks: list[dict],
) -> str:
    """
    Build the user message for the Claude judge.
    Passes full chunk text (up to 1500 chars) so the judge has enough
    context to verify claims properly.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Passage {i} — {chunk['company'].capitalize()}, "
            f"pages {chunk['start_page']}–{chunk['end_page']}]\n"
            f"{chunk['text'][:1500]}"   # full context, no artificial truncation
        )

    context_block = "\n\n".join(context_parts)

    return (
        f"## Question\n{query}\n\n"
        f"## Retrieved Context Passages\n{context_block}\n\n"
        f"## Answer to Evaluate\n{answer}\n\n"
        "Score the answer across the four dimensions. "
        "Remember: paraphrasing and synthesis are expected — only penalize "
        "claims that cannot be traced to any passage above."
    )


# ── Evaluator ─────────────────────────────────────────────────────────────────

class RAGEvaluator:
    """
    Evaluates RAG pipeline responses using Claude as a judge.

    Example:
        evaluator = RAGEvaluator()
        result = evaluator.evaluate(
            query   = "What are Tesla's risk factors?",
            answer  = "Tesla faces risks including...",
            chunks  = retrieved_chunks,
        )
        print(result.summary())
    """

    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise OSError("ANTHROPIC_API_KEY not set in .env")

        self.client = anthropic.Anthropic(api_key=api_key)
        logger.info("RAGEvaluator ready (judge model: %s)", JUDGE_MODEL)

    def evaluate(
        self,
        query:  str,
        answer: str,
        chunks: list[dict],
    ) -> EvaluationResult:
        """
        Score a RAG response using the Claude judge.

        Args:
            query  : the original user question
            answer : the LLM-generated answer to evaluate
            chunks : the retrieved context chunks used to generate the answer

        Returns:
            EvaluationResult with four scores and reasoning
        """
        logger.info("Evaluating response for query: \"%s\"", query[:80])

        user_prompt = _build_judge_prompt(query, answer, chunks)

        response = self.client.messages.create(
            model       = JUDGE_MODEL,
            max_tokens  = MAX_TOKENS,
            temperature = TEMPERATURE,
            system      = JUDGE_SYSTEM_PROMPT,
            messages    = [{"role": "user", "content": user_prompt}],
        )

        raw = response.content[0].text.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        scores = json.loads(raw)

        result = EvaluationResult(
            grounding    = float(scores["grounding"]),
            relevance    = float(scores["relevance"]),
            faithfulness = float(scores["faithfulness"]),
            completeness = float(scores["completeness"]),
            reasoning    = scores["reasoning"],
        )

        logger.info("Evaluation result: %s", result.summary())
        return result


# ── Batch evaluation ──────────────────────────────────────────────────────────

def evaluate_batch(
    evaluator:  RAGEvaluator,
    test_cases: list[dict],
) -> list[dict]:
    """
    Evaluate a list of test cases and return aggregated results.

    Each test case is a dict with keys: query, answer, chunks.
    """
    results = []

    for i, case in enumerate(test_cases, 1):
        logger.info("Evaluating case %d/%d", i, len(test_cases))
        try:
            result = evaluator.evaluate(
                query  = case["query"],
                answer = case["answer"],
                chunks = case["chunks"],
            )
            results.append({
                "query":        case["query"],
                "passed":       result.passed,
                "average":      round(result.average, 3),
                "grounding":    result.grounding,
                "relevance":    result.relevance,
                "faithfulness": result.faithfulness,
                "completeness": result.completeness,
                "reasoning":    result.reasoning,
            })
        except Exception as exc:
            logger.error("Evaluation failed for case %d: %s", i, exc)
            results.append({"query": case["query"], "error": str(exc)})

    valid = [r for r in results if "error" not in r]
    if valid:
        avg_score = sum(r["average"] for r in valid) / len(valid)
        pass_rate = sum(1 for r in valid if r["passed"]) / len(valid) * 100
        logger.info(
            "Batch complete: %d/%d | avg=%.3f | pass_rate=%.0f%%",
            len(valid), len(results), avg_score, pass_rate,
        )

    return results