"""
test_evaluation.py
------------------
Unit tests for guardrails and the RAG evaluator.

All external API calls (Claude judge, OpenAI) are mocked so tests run
instantly without consuming any API credits.

Run with:
    pytest tests/test_evaluation.py -v
"""

from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.guardrails import (
    GuardrailResult,
    PostGuardrail,
    PreGuardrail,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def good_chunks() -> list[dict]:
    """High-confidence chunks that should pass all guardrails."""
    return [
        {
            "company": "tesla",
            "source": "tesla_10k.htm",
            "chunk_id": 0,
            "start_page": 10,
            "end_page": 11,
            "text": "Tesla faces macroeconomic risks including inflation.",
            "rerank_score": 0.85,
        },
        {
            "company": "tesla",
            "source": "tesla_10k.htm",
            "chunk_id": 1,
            "start_page": 12,
            "end_page": 13,
            "text": "Key personnel risk: the company depends on Elon Musk.",
            "rerank_score": 0.72,
        },
    ]


@pytest.fixture()
def low_score_chunks() -> list[dict]:
    """Chunks with very low scores that should fail the confidence check."""
    return [
        {
            "company": "apple",
            "source": "apple_10k.htm",
            "chunk_id": 2,
            "start_page": 5,
            "end_page": 6,
            "text": "Apple designs iPhones and Macs.",
            "rerank_score": 0.0005,
        },
    ]


@pytest.fixture()
def good_answer() -> str:
    return (
        "Tesla faces several key risk factors:\n"
        "- Macroeconomic risks including inflation [Tesla | pages 10–11].\n"
        "- Key personnel dependency on Elon Musk [Tesla | pages 12–13]."
    )


@pytest.fixture()
def hallucinated_answer() -> str:
    return (
        "Based on my knowledge, Tesla generally faces supply chain issues. "
        "I believe the company also has significant debt. "
        "This is a well-known fact in the industry."
    )


# ── PreGuardrail.check_on_topic ───────────────────────────────────────────────


class TestCheckOnTopic:
    def test_passes_for_company_name(self):
        assert PreGuardrail.check_on_topic("What are Tesla's risk factors?").passed

    def test_passes_for_financial_term(self):
        assert PreGuardrail.check_on_topic("What is the revenue growth?").passed

    def test_passes_for_microsoft(self):
        assert PreGuardrail.check_on_topic("How is Azure performing?").passed

    def test_fails_for_off_topic(self):
        result = PreGuardrail.check_on_topic("What is the weather in Madrid?")
        assert not result.passed
        assert result.reason == "off_topic"
        assert len(result.message) > 20

    def test_fails_for_general_chat(self):
        result = PreGuardrail.check_on_topic("Tell me a joke")
        assert not result.passed

    def test_message_is_helpful(self):
        result = PreGuardrail.check_on_topic("Who won the football match?")
        assert "Tesla" in result.message or "Apple" in result.message


# ── PreGuardrail.check_has_chunks ─────────────────────────────────────────────


class TestCheckHasChunks:
    def test_passes_with_chunks(self, good_chunks):
        assert PreGuardrail.check_has_chunks(good_chunks).passed

    def test_fails_with_empty_list(self):
        result = PreGuardrail.check_has_chunks([])
        assert not result.passed
        assert result.reason == "no_chunks"

    def test_message_is_actionable(self):
        result = PreGuardrail.check_has_chunks([])
        assert len(result.message) > 20


# ── PreGuardrail.check_confidence ─────────────────────────────────────────────


class TestCheckConfidence:
    def test_passes_high_rerank_score(self, good_chunks):
        assert PreGuardrail.check_confidence(good_chunks).passed

    def test_fails_low_rerank_score(self, low_score_chunks):
        result = PreGuardrail.check_confidence(low_score_chunks)
        assert not result.passed
        assert "low_confidence" in result.reason

    def test_uses_faiss_score_when_no_rerank(self):
        chunks = [{"faiss_score": 0.05, "company": "apple", "chunk_id": 0, "text": "x"}]
        result = PreGuardrail.check_confidence(chunks)
        assert not result.passed

    def test_passes_good_faiss_score(self):
        chunks = [{"faiss_score": 0.80, "company": "tesla", "chunk_id": 0, "text": "x"}]
        assert PreGuardrail.check_confidence(chunks).passed

    def test_uses_best_score_not_average(self):
        chunks = [
            {"rerank_score": 0.001, "company": "tesla", "chunk_id": 0, "text": "x"},
            {"rerank_score": 0.900, "company": "tesla", "chunk_id": 1, "text": "y"},
        ]
        # Best score (0.9) should pass even though first chunk is low
        assert PreGuardrail.check_confidence(chunks).passed


# ── PreGuardrail.run_all ──────────────────────────────────────────────────────


class TestPreGuardrailRunAll:
    def test_passes_valid_query_and_chunks(self, good_chunks):
        result = PreGuardrail.run_all("What are Tesla's risk factors?", good_chunks)
        assert result.passed

    def test_fails_off_topic_before_checking_chunks(self):
        # Off-topic check should fire even if chunks are good
        result = PreGuardrail.run_all("Tell me a joke", [])
        assert not result.passed
        assert result.reason == "off_topic"

    def test_fails_on_empty_chunks(self):
        result = PreGuardrail.run_all("What are Apple revenues?", [])
        assert not result.passed

    def test_fails_on_low_confidence(self, low_score_chunks):
        result = PreGuardrail.run_all("Apple revenue", low_score_chunks)
        assert not result.passed


# ── PostGuardrail.check_no_hallucination_phrases ─────────────────────────────


class TestCheckNoHallucinationPhrases:
    def test_passes_clean_answer(self, good_answer):
        assert PostGuardrail.check_no_hallucination_phrases(good_answer).passed

    def test_fails_on_based_on_my_knowledge(self, hallucinated_answer):
        result = PostGuardrail.check_no_hallucination_phrases(hallucinated_answer)
        assert not result.passed
        assert "hallucination_phrases" in result.reason

    def test_fails_on_i_believe(self):
        result = PostGuardrail.check_no_hallucination_phrases("I believe this is correct.")
        assert not result.passed

    def test_case_insensitive(self):
        result = PostGuardrail.check_no_hallucination_phrases("BASED ON MY KNOWLEDGE this is true.")
        assert not result.passed


# ── PostGuardrail.check_has_citations ────────────────────────────────────────


class TestCheckHasCitations:
    def test_passes_with_citations(self, good_answer):
        assert PostGuardrail.check_has_citations(good_answer).passed

    def test_fails_without_citations(self):
        result = PostGuardrail.check_has_citations(
            "Tesla faces many risks in the current market environment."
        )
        assert not result.passed
        assert result.reason == "no_citations"

    def test_passes_all_three_companies(self):
        for company in ["Tesla", "Apple", "Microsoft"]:
            answer = f"Revenue grew significantly [{company} | pages 1–2]."
            assert PostGuardrail.check_has_citations(answer).passed

    def test_case_insensitive_company(self):
        assert PostGuardrail.check_has_citations("Revenue grew [tesla | pages 1–2].").passed


# ── PostGuardrail.run_all ─────────────────────────────────────────────────────


class TestPostGuardrailRunAll:
    def test_returns_list(self, good_answer):
        results = PostGuardrail.run_all(good_answer)
        assert isinstance(results, list)
        assert len(results) == 2  # two checks currently

    def test_all_pass_for_good_answer(self, good_answer):
        results = PostGuardrail.run_all(good_answer)
        assert all(r.passed for r in results)

    def test_detects_hallucination_in_batch(self, hallucinated_answer):
        results = PostGuardrail.run_all(hallucinated_answer)
        failed = [r for r in results if not r.passed]
        assert len(failed) >= 1


# ── GuardrailResult ───────────────────────────────────────────────────────────


class TestGuardrailResult:
    def test_bool_true_when_passed(self):
        assert bool(GuardrailResult(passed=True))

    def test_bool_false_when_failed(self):
        assert not bool(GuardrailResult(passed=False, reason="test"))

    def test_repr_contains_status(self):
        r = GuardrailResult(passed=True)
        assert "PASS" in repr(r)

        r = GuardrailResult(passed=False, reason="low_confidence")
        assert "FAIL" in repr(r)


# ── EvaluationResult ──────────────────────────────────────────────────────────


class TestEvaluationResult:
    def test_passed_when_all_above_threshold(self):
        from src.evaluation.evaluator import EvaluationResult

        result = EvaluationResult(
            grounding=0.90,
            relevance=0.85,
            faithfulness=0.92,
            completeness=0.75,
            reasoning={},
        )
        assert result.passed

    def test_failed_when_one_below_threshold(self):
        from src.evaluation.evaluator import EvaluationResult

        result = EvaluationResult(
            grounding=0.30,  # below 0.70 threshold
            relevance=0.85,
            faithfulness=0.92,
            completeness=0.75,
            reasoning={},
        )
        assert not result.passed

    def test_average_computed_correctly(self):
        from src.evaluation.evaluator import EvaluationResult

        result = EvaluationResult(
            grounding=1.0,
            relevance=0.8,
            faithfulness=0.6,
            completeness=0.4,
            reasoning={},
        )
        assert abs(result.average - 0.7) < 1e-6

    def test_summary_contains_all_dimensions(self):
        from src.evaluation.evaluator import EvaluationResult

        result = EvaluationResult(
            grounding=0.9,
            relevance=0.8,
            faithfulness=0.85,
            completeness=0.7,
            reasoning={},
        )
        summary = result.summary()
        assert "grounding" in summary
        assert "relevance" in summary
        assert "faithfulness" in summary
        assert "completeness" in summary

    def test_to_dict_returns_dict(self):
        from src.evaluation.evaluator import EvaluationResult

        result = EvaluationResult(
            grounding=0.9,
            relevance=0.8,
            faithfulness=0.85,
            completeness=0.7,
            reasoning={"grounding": "Good"},
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "grounding" in d


# ── RAGEvaluator (mocked) ─────────────────────────────────────────────────────


class TestRAGEvaluator:
    def _make_evaluator(self):
        with (
            patch("src.evaluation.evaluator.anthropic.Anthropic"),
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}),
        ):
            from src.evaluation.evaluator import RAGEvaluator

            return RAGEvaluator()

    def test_evaluate_parses_json_response(self, good_chunks, good_answer):
        evaluator = self._make_evaluator()

        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text="""{
            "grounding": 0.95,
            "relevance": 0.88,
            "faithfulness": 0.92,
            "completeness": 0.80,
            "reasoning": {
                "grounding": "All claims are sourced.",
                "relevance": "Directly answers the question.",
                "faithfulness": "No hallucinations detected.",
                "completeness": "Covers the main points."
            }
        }"""
            )
        ]

        evaluator.client.messages.create = MagicMock(return_value=mock_response)

        result = evaluator.evaluate("What are Tesla's risks?", good_answer, good_chunks)

        assert result.grounding == 0.95
        assert result.relevance == 0.88
        assert result.faithfulness == 0.92
        assert result.completeness == 0.80
        assert result.passed

    def test_evaluate_handles_markdown_fences(self, good_chunks, good_answer):
        evaluator = self._make_evaluator()

        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text="""```json
{
    "grounding": 0.80,
    "relevance": 0.75,
    "faithfulness": 0.85,
    "completeness": 0.70,
    "reasoning": {
        "grounding": "ok", "relevance": "ok",
        "faithfulness": "ok", "completeness": "ok"
    }
}
```"""
            )
        ]
        evaluator.client.messages.create = MagicMock(return_value=mock_response)

        result = evaluator.evaluate("question", good_answer, good_chunks)
        assert result.grounding == 0.80
