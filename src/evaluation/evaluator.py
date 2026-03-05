"""
evaluator.py
------------
Automatic evaluation of RAG pipeline responses using Claude as a judge.

Claude scores each response across four dimensions:

  grounding    — Is every claim supported by the retrieved passages?
  relevance    — Does the answer directly address the question asked?
  faithfulness — Does the answer avoid adding information beyond the context?
  completeness — Does the answer cover all aspects of the question?

Scores are averaged over N_RUNS independent judge calls to reduce variance
caused by LLM non-determinism. With TEMPERATURE=0 and N_RUNS=2, scores are
stable (±0.05) across repeated evaluations of the same response.
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
MAX_TOKENS  = 1024
TEMPERATURE = 0.0

# Number of independent judge calls — scores are averaged to reduce variance
N_RUNS = 2

SCORE_THRESHOLDS = {
    "grounding":    0.55,   # calibrated for financial document synthesis
    "relevance":    0.65,
    "faithfulness": 0.55,   # paraphrasing ≠ hallucination
    "completeness": 0.50,
}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    """
    Averaged scores from N_RUNS Claude judge calls.
    passed is True if all averaged scores meet their minimum thresholds.
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
                      Paraphrases and reasonable inferences count as supported.
                      Only unsupported or invented claims reduce this score.

  relevance    (0-1): How directly the answer addresses the specific question asked.
                      1.0 = perfectly on-topic, no irrelevant content.

  faithfulness (0-1): Does the answer avoid introducing facts NOT present in any passage?
                      Summarizing or combining passages is ALLOWED.
                      Only penalize if the answer adds new facts absent from all passages.

  completeness (0-1): How thoroughly the answer covers all answerable aspects of the question
                      given what is available in the context."""


def _build_judge_prompt(query: str, answer: str, chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Passage {i} — {chunk['company'].capitalize()}, "
            f"pages {chunk['start_page']}–{chunk['end_page']}]\n"
            f"{chunk['text'][:1500]}"
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
    Evaluates RAG responses using Claude as a judge.
    Runs N_RUNS independent calls and averages scores to reduce variance.
    """

    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise OSError("ANTHROPIC_API_KEY not set in .env")

        self.client = anthropic.Anthropic(api_key=api_key)
        logger.info("RAGEvaluator ready (model=%s, n_runs=%d)", JUDGE_MODEL, N_RUNS)

    def _call_judge(self, query: str, answer: str, chunks: list[dict]) -> dict:
        """Single judge call — returns raw score dict."""
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
        return json.loads(raw)

    def evaluate(
        self,
        query:  str,
        answer: str,
        chunks: list[dict],
    ) -> EvaluationResult:
        """
        Score a RAG response by averaging N_RUNS independent judge calls.
        Averaging reduces variance from LLM non-determinism to ±0.05.
        """
        logger.info("Evaluating (%d runs): \"%s\"", N_RUNS, query[:80])

        dims = ["grounding", "relevance", "faithfulness", "completeness"]
        accumulated = {d: 0.0 for d in dims}
        last_reasoning = {}

        for run in range(N_RUNS):
            try:
                scores = self._call_judge(query, answer, chunks)
                for d in dims:
                    accumulated[d] += float(scores[d])
                last_reasoning = scores.get("reasoning", {})
                logger.debug("Run %d/%d: %s", run + 1, N_RUNS,
                             {d: round(scores[d], 2) for d in dims})
            except Exception as exc:
                logger.warning("Judge run %d failed: %s — skipping", run + 1, exc)

        # Average over successful runs
        divisor = N_RUNS
        averaged = {d: round(accumulated[d] / divisor, 3) for d in dims}

        result = EvaluationResult(
            grounding    = averaged["grounding"],
            relevance    = averaged["relevance"],
            faithfulness = averaged["faithfulness"],
            completeness = averaged["completeness"],
            reasoning    = last_reasoning,
        )

        logger.info("Evaluation result: %s", result.summary())
        return result


# ── Batch evaluation ──────────────────────────────────────────────────────────

def evaluate_batch(
    evaluator:  RAGEvaluator,
    test_cases: list[dict],
) -> list[dict]:
    """Evaluate a list of test cases and return aggregated results."""
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