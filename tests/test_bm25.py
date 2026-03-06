"""
test_bm25.py
------------
Unit tests for the BM25 lexical search index.

Run with:
    pytest tests/test_bm25.py -v
"""

import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def sample_metadata() -> list[dict]:
    return [
        {
            "chunk_id": 0,
            "company": "tesla",
            "year": 2024,
            "source": "tesla_10k_2024.htm",
            "start_page": 10,
            "end_page": 11,
            "text": "Tesla faces macroeconomic risks including inflation and "
            "interest rate increases reducing demand for electric vehicles.",
        },
        {
            "chunk_id": 1,
            "company": "tesla",
            "year": 2023,
            "source": "tesla_10k_2023.htm",
            "start_page": 16,
            "end_page": 17,
            "text": "Tesla supply chain disruptions and semiconductor shortages "
            "impact production capacity and delivery timelines.",
        },
        {
            "chunk_id": 2,
            "company": "apple",
            "year": 2024,
            "source": "apple_10k_2024.htm",
            "start_page": 42,
            "end_page": 43,
            "text": "Apple iPhone revenue reached $200B in fiscal 2024 driven "
            "by strong iPhone 15 demand and services growth.",
        },
        {
            "chunk_id": 3,
            "company": "microsoft",
            "year": 2024,
            "source": "microsoft_10k_2024.htm",
            "start_page": 67,
            "end_page": 68,
            "text": "Microsoft Azure revenue grew 29% year-over-year in fiscal 2024 "
            "driven by AI services and cloud infrastructure.",
        },
        {
            "chunk_id": 4,
            "company": "microsoft",
            "year": 2022,
            "source": "microsoft_10k_2022.htm",
            "start_page": 47,
            "end_page": 48,
            "text": "Microsoft EBITDA margin improved as commercial cloud revenue "
            "grew 32% in fiscal 2022.",
        },
    ]


# ── build_bm25_index ──────────────────────────────────────────────────────────


class TestBuildBM25Index:
    def test_returns_bm25_object(self, sample_metadata):
        from rank_bm25 import BM25Okapi

        from src.retrieval.bm25_index import build_bm25_index

        bm25 = build_bm25_index(sample_metadata)
        assert isinstance(bm25, BM25Okapi)

    def test_index_has_correct_corpus_size(self, sample_metadata):
        from src.retrieval.bm25_index import build_bm25_index

        bm25 = build_bm25_index(sample_metadata)
        assert bm25.corpus_size == len(sample_metadata)

    def test_empty_metadata_raises(self):
        """BM25Okapi does not support empty corpus — verify it raises."""
        from src.retrieval.bm25_index import build_bm25_index

        with pytest.raises((ZeroDivisionError, ValueError)):
            build_bm25_index([])


# ── bm25_search ───────────────────────────────────────────────────────────────


class TestBM25Search:
    def test_returns_list(self, sample_metadata):
        from src.retrieval.bm25_index import bm25_search, build_bm25_index

        bm25 = build_bm25_index(sample_metadata)
        results = bm25_search("Tesla risks", bm25, sample_metadata, top_k=3)
        assert isinstance(results, list)

    def test_top_k_respected(self, sample_metadata):
        from src.retrieval.bm25_index import bm25_search, build_bm25_index

        bm25 = build_bm25_index(sample_metadata)
        results = bm25_search("revenue", bm25, sample_metadata, top_k=2)
        assert len(results) <= 2

    def test_results_have_bm25_score(self, sample_metadata):
        from src.retrieval.bm25_index import bm25_search, build_bm25_index

        bm25 = build_bm25_index(sample_metadata)
        results = bm25_search("Azure cloud", bm25, sample_metadata, top_k=3)
        for chunk in results:
            assert "bm25_score" in chunk
            assert isinstance(chunk["bm25_score"], float)

    def test_relevant_result_ranked_first(self, sample_metadata):
        """A query with exact keywords should surface the matching chunk first."""
        from src.retrieval.bm25_index import bm25_search, build_bm25_index

        bm25 = build_bm25_index(sample_metadata)
        results = bm25_search("EBITDA margin", bm25, sample_metadata, top_k=5)
        assert len(results) > 0
        assert results[0]["company"] == "microsoft"
        assert results[0]["year"] == 2022

    def test_company_filter(self, sample_metadata):
        from src.retrieval.bm25_index import bm25_search, build_bm25_index

        bm25 = build_bm25_index(sample_metadata)
        results = bm25_search(
            "revenue",
            bm25,
            sample_metadata,
            top_k=5,
            company_filter="apple",
        )
        for chunk in results:
            assert chunk["company"] == "apple"

    def test_year_filter(self, sample_metadata):
        from src.retrieval.bm25_index import bm25_search, build_bm25_index

        bm25 = build_bm25_index(sample_metadata)
        results = bm25_search(
            "cloud revenue",
            bm25,
            sample_metadata,
            top_k=5,
            year_filter=2024,
        )
        for chunk in results:
            assert chunk["year"] == 2024

    def test_no_results_for_unrelated_query(self, sample_metadata):
        """A query with zero BM25 score should return empty or low-score results."""
        from src.retrieval.bm25_index import bm25_search, build_bm25_index

        bm25 = build_bm25_index(sample_metadata)
        results = bm25_search(
            "zzzzzzz xxxxxxxxxxx",
            bm25,
            sample_metadata,
            top_k=3,
        )
        # Either empty or all scores are zero
        for chunk in results:
            assert chunk["bm25_score"] == 0.0

    def test_returns_chunk_metadata_intact(self, sample_metadata):
        """Results should contain all original metadata fields."""
        from src.retrieval.bm25_index import bm25_search, build_bm25_index

        bm25 = build_bm25_index(sample_metadata)
        results = bm25_search("Tesla semiconductor", bm25, sample_metadata, top_k=1)
        if results:
            chunk = results[0]
            assert "company" in chunk
            assert "year" in chunk
            assert "text" in chunk
            assert "start_page" in chunk
