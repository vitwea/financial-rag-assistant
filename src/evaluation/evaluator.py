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
    print(scores)
    # EvaluationResult(grounding=0.95, relevance=0.88, ...)
"""

import json
import os
from dataclasses import dataclass, asdict

import anthropic
from dotenv import load_dotenv

from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

# Claude is used as the judge — better reasoning and instruction-following
# than GPT-4o-mini for structured evaluation tasks
JUDGE_MODEL  = "claude-haiku-4-5-20251001"   # fast and cheap, ideal for evaluation
MAX_TOKENS   = 512
TEMPERATURE  = 0.0    # deterministic scoring

# Minimum acceptable score for each dimension (used for pass/fail reporting)
SCORE_THRESHOLDS = {
    "grounding":    0.70,
    "relevance":    0.70,
    "faithfulness": 0.80,
    "completeness": 0.60,
}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    """
    Scores returned by the Claude judge for a single RAG response.

    All scores are in the range [0.0, 1.0].
    reasoning contains Claude's explanation for each score.
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


# ── Prompt builder ────────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of RAG (Retrieval-Augmented Generation)
systems specializing in financial document analysis.

Your task is to score an AI-generated answer based on the retrieved context passages
that were provided to the AI. Score objectively and strictly.

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
  grounding    (0-1): Fraction of claims in the answer that are directly supported
                      by the retrieved passages. 1.0 = every claim has a source.
  relevance    (0-1): How well the answer addresses the specific question asked.
                      1.0 = perfectly on-topic, no irrelevant content.
  faithfulness (0-1): Degree to which the answer stays within the context.
                      1.0 = no information beyond what the passages contain.
  completeness (0-1): How thoroughly the answer covers all aspects of the question
                      given what is available in the context.
                      1.0 = all answerable aspects are addressed."""


def _build_judge_prompt(
    query: str,
    answer: str,
    chunks: list[dict],
) -> str:
    """
    Build the user message for the Claude judge.
    Includes the question, the retrieved passages, and the answer to evaluate.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Passage {i} — {chunk['company'].capitalize()}, "
            f"pages {chunk['start_page']}–{chunk['end_page']}]\n"
            f"{chunk['text'][:600]}..."
        )

    context_block = "\n\n".join(context_parts)

    return (
        f"## Question\n{query}\n\n"
        f"## Retrieved Context Passages\n{context_block}\n\n"
        f"## Answer to Evaluate\n{answer}\n\n"
        "Please score the answer according to the four dimensions defined in your instructions."
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
            raise EnvironmentError("ANTHROPIC_API_KEY not set in .env")

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

        # Strip markdown code fences if present
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
    evaluator: RAGEvaluator,
    test_cases: list[dict],
) -> list[dict]:
    """
    Evaluate a list of test cases and return aggregated results.

    Each test case is a dict with keys: query, answer, chunks.
    Returns a list of dicts with query + evaluation scores.

    Example:
        cases = [
            {"query": "Tesla risks?", "answer": "...", "chunks": [...]},
        ]
        results = evaluate_batch(evaluator, cases)
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
                "query":       case["query"],
                "passed":      result.passed,
                "average":     round(result.average, 3),
                "grounding":   result.grounding,
                "relevance":   result.relevance,
                "faithfulness":result.faithfulness,
                "completeness":result.completeness,
                "reasoning":   result.reasoning,
            })
        except Exception as exc:
            logger.error("Evaluation failed for case %d: %s", i, exc)
            results.append({"query": case["query"], "error": str(exc)})

    # Print aggregate summary
    valid = [r for r in results if "error" not in r]
    if valid:
        avg_score = sum(r["average"] for r in valid) / len(valid)
        pass_rate = sum(1 for r in valid if r["passed"]) / len(valid) * 100
        logger.info(
            "Batch evaluation complete: %d/%d cases | avg=%.3f | pass_rate=%.0f%%",
            len(valid), len(results), avg_score, pass_rate,
        )

    return results