"""
test_pipeline.py
----------------
Unit tests for the retrieval and RAG pipeline modules.

All external services (FAISS, OpenAI, Cohere) are mocked so tests
run instantly without API keys or a built index.

Run with:
    pytest tests/test_pipeline.py -v
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture()
def sample_chunks() -> list[dict]:
    """A minimal list of chunk records simulating a loaded index."""
    return [
        {
            "doc_id": 0, "chunk_id": 0,
            "company": "tesla", "year": 2024, "source": "tesla_10k_2024.htm",
            "text": "Tesla faces risks from macroeconomic conditions including "
                    "inflation and interest rate increases which may reduce consumer "
                    "demand for electric vehicles.",
            "start_page": 10, "end_page": 11, "word_count": 30,
            "faiss_score": 0.91,
        },
        {
            "doc_id": 1, "chunk_id": 1,
            "company": "tesla", "year": 2023, "source": "tesla_10k_2023.htm",
            "text": "Tesla faces supply chain disruptions and semiconductor shortages "
                    "which impact production capacity and delivery timelines significantly.",
            "start_page": 16, "end_page": 17, "word_count": 28,
            "faiss_score": 0.87,
        },
        {
            "doc_id": 2, "chunk_id": 2,
            "company": "apple", "year": 2025, "source": "apple_10k_2025.htm",
            "text": "Apple Services segment generated revenue of $85 billion in "
                    "fiscal 2025, driven by App Store, Apple Music, and iCloud.",
            "start_page": 42, "end_page": 43, "word_count": 25,
            "faiss_score": 0.85,
        },
        {
            "doc_id": 3, "chunk_id": 3,
            "company": "microsoft", "year": 2025, "source": "microsoft_10k_2025.htm",
            "text": "Microsoft Azure revenue grew 29% year-over-year in fiscal 2025, "
                    "driven by AI services and cloud infrastructure demand.",
            "start_page": 67, "end_page": 68, "word_count": 28,
            "faiss_score": 0.78,
        },
        {
            "doc_id": 4, "chunk_id": 4,
            "company": "microsoft", "year": 2022, "source": "microsoft_10k_2022.htm",
            "text": "Microsoft cloud services revenue grew 32% in fiscal 2022 "
                    "as enterprises accelerated digital transformation initiatives.",
            "start_page": 47, "end_page": 48, "word_count": 26,
            "faiss_score": 0.72,
        },
    ]


# ── build_prompt ──────────────────────────────────────────────────────────────

class TestBuildPrompt:
    def test_contains_query(self, sample_chunks):
        from src.pipeline.pipeline import build_prompt
        query = "What are Tesla risk factors?"
        result = build_prompt(query, sample_chunks[:1])
        assert query in result

    def test_contains_company_name(self, sample_chunks):
        from src.pipeline.pipeline import build_prompt
        result = build_prompt("Any question?", sample_chunks)
        assert "Tesla" in result
        assert "Apple" in result
        assert "Microsoft" in result

    def test_contains_page_references(self, sample_chunks):
        from src.pipeline.pipeline import build_prompt
        result = build_prompt("Any question?", sample_chunks[:1])
        assert "10" in result
        assert "11" in result

    def test_passage_numbering(self, sample_chunks):
        from src.pipeline.pipeline import build_prompt
        result = build_prompt("Any question?", sample_chunks)
        assert "Passage 1" in result
        assert "Passage 2" in result
        assert "Passage 3" in result

    def test_empty_chunks(self):
        from src.pipeline.pipeline import build_prompt
        result = build_prompt("question?", [])
        assert "question?" in result
        assert "[Context]" in result


# ── format_sources ────────────────────────────────────────────────────────────

class TestFormatSources:
    def test_lists_all_companies(self, sample_chunks):
        from src.pipeline.pipeline import format_sources
        result = format_sources(sample_chunks)
        assert "Tesla" in result
        assert "Apple" in result
        assert "Microsoft" in result

    def test_deduplicates_same_pages(self, sample_chunks):
        from src.pipeline.pipeline import format_sources
        doubled = sample_chunks[:1] * 3
        result = format_sources(doubled)
        assert result.count("pages 10") == 1

    def test_includes_page_range(self, sample_chunks):
        from src.pipeline.pipeline import format_sources
        result = format_sources(sample_chunks[:1])
        assert "10" in result
        assert "11" in result

    def test_returns_string(self, sample_chunks):
        from src.pipeline.pipeline import format_sources
        assert isinstance(format_sources(sample_chunks), str)

    def test_includes_year_in_source_filename(self, sample_chunks):
        from src.pipeline.pipeline import format_sources
        result = format_sources(sample_chunks[:1])
        assert "2024" in result


# ── RAGPipeline ───────────────────────────────────────────────────────────────

class TestRAGPipeline:
    """Tests for RAGPipeline.ask() with all external calls mocked."""

    def _make_pipeline(self, sample_chunks):
        """Build a RAGPipeline with mocked index, metadata and bm25."""
        with patch("src.pipeline.pipeline.load_index") as mock_load, \
             patch("src.pipeline.pipeline.OpenAI"), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            # load_index() now returns a 3-tuple: (faiss_index, metadata, bm25)
            mock_load.return_value = (MagicMock(), sample_chunks, MagicMock())
            from src.pipeline.pipeline import RAGPipeline
            pipeline = RAGPipeline()
        return pipeline

    def test_ask_returns_required_keys(self, sample_chunks):
        pipeline = self._make_pipeline(sample_chunks)

        with patch("src.pipeline.pipeline.retrieve") as mock_retrieve, \
             patch("src.pipeline.pipeline.call_llm") as mock_llm:
            mock_retrieve.return_value = sample_chunks[:2]
            mock_llm.return_value = "Tesla faces macroeconomic risks [Tesla | pages 10–11]."

            result = pipeline.ask("What are Tesla risk factors?")

        assert "query"       in result
        assert "answer"      in result
        assert "sources"     in result
        assert "chunks_used" in result

    def test_ask_passes_query_through(self, sample_chunks):
        pipeline = self._make_pipeline(sample_chunks)
        query = "What is Apple Services revenue breakdown?"

        with patch("src.pipeline.pipeline.retrieve") as mock_retrieve, \
             patch("src.pipeline.pipeline.call_llm") as mock_llm:
            mock_retrieve.return_value = sample_chunks[2:3]
            mock_llm.return_value = "Apple Services generated $85B."

            result = pipeline.ask(query)

        assert result["query"] == query

    def test_ask_handles_no_chunks(self, sample_chunks):
        """When retrieve returns empty, pipeline should return a no-passages message."""
        pipeline = self._make_pipeline(sample_chunks)

        with patch("src.pipeline.pipeline.retrieve") as mock_retrieve:
            mock_retrieve.return_value = []
            result = pipeline.ask("What are Tesla capital expenditure amounts?")

        assert result["chunks_used"] == []
        assert len(result["answer"]) > 0

    def test_company_filter_forwarded(self, sample_chunks):
        pipeline = self._make_pipeline(sample_chunks)

        with patch("src.pipeline.pipeline.retrieve") as mock_retrieve, \
             patch("src.pipeline.pipeline.call_llm") as mock_llm:
            mock_retrieve.return_value = sample_chunks[:1]
            mock_llm.return_value = "Tesla risk answer."

            pipeline.ask("What are Tesla main risk factors?", company_filter="tesla")

            call_kwargs = mock_retrieve.call_args
            assert call_kwargs is not None, "retrieve() was never called"
            assert call_kwargs.kwargs.get("company_filter") == "tesla" or \
                   (len(call_kwargs.args) >= 4 and call_kwargs.args[3] == "tesla")

    def test_year_filter_forwarded(self, sample_chunks):
        pipeline = self._make_pipeline(sample_chunks)

        with patch("src.pipeline.pipeline.retrieve") as mock_retrieve, \
             patch("src.pipeline.pipeline.call_llm") as mock_llm:
            mock_retrieve.return_value = sample_chunks[:1]
            mock_llm.return_value = "Microsoft 2022 risks."

            pipeline.ask("What are Microsoft risk factors?", year_filter=2022)

            call_kwargs = mock_retrieve.call_args
            assert call_kwargs is not None, "retrieve() was never called"
            assert call_kwargs.kwargs.get("year_filter") == 2022 or \
                   (len(call_kwargs.args) >= 5 and call_kwargs.args[4] == 2022)

    def test_company_and_year_filter_combined(self, sample_chunks):
        pipeline = self._make_pipeline(sample_chunks)

        with patch("src.pipeline.pipeline.retrieve") as mock_retrieve, \
             patch("src.pipeline.pipeline.call_llm") as mock_llm:
            mock_retrieve.return_value = sample_chunks[:1]
            mock_llm.return_value = "Microsoft 2022 cloud revenue."

            pipeline.ask(
                "What is Microsoft cloud revenue?",
                company_filter="microsoft",
                year_filter=2022,
            )

            call_kwargs = mock_retrieve.call_args
            assert call_kwargs is not None, "retrieve() was never called"


# ── FAISS search ──────────────────────────────────────────────────────────────

class TestFAISSSearch:
    def _make_index(self, n_vectors: int = 5, dim: int = 8):
        import faiss as _faiss
        vectors = np.random.rand(n_vectors, dim).astype("float32")
        _faiss.normalize_L2(vectors)
        index = _faiss.IndexFlatIP(dim)
        index.add(vectors)
        return index, vectors

    def test_returns_list(self, sample_chunks):
        from src.retrieval.retriever import faiss_search

        index, _ = self._make_index(n_vectors=len(sample_chunks), dim=8)

        with patch("src.retrieval.retriever.embed_query") as mock_embed:
            q_vec = np.random.rand(1, 8).astype("float32")
            import faiss as _faiss
            _faiss.normalize_L2(q_vec)
            mock_embed.return_value = q_vec
            results = faiss_search("Tesla risk factors", index, sample_chunks, top_k=3)

        assert isinstance(results, list)
        assert len(results) <= 3

    def test_company_filter_applied(self, sample_chunks):
        from src.retrieval.retriever import faiss_search

        index, _ = self._make_index(n_vectors=len(sample_chunks), dim=8)

        with patch("src.retrieval.retriever.embed_query") as mock_embed:
            q_vec = np.random.rand(1, 8).astype("float32")
            import faiss as _faiss
            _faiss.normalize_L2(q_vec)
            mock_embed.return_value = q_vec
            results = faiss_search(
                "Apple revenue", index, sample_chunks,
                top_k=5, company_filter="apple",
            )

        for chunk in results:
            assert chunk["company"] == "apple"

    def test_year_filter_applied(self, sample_chunks):
        from src.retrieval.retriever import faiss_search

        index, _ = self._make_index(n_vectors=len(sample_chunks), dim=8)

        with patch("src.retrieval.retriever.embed_query") as mock_embed:
            q_vec = np.random.rand(1, 8).astype("float32")
            import faiss as _faiss
            _faiss.normalize_L2(q_vec)
            mock_embed.return_value = q_vec
            results = faiss_search(
                "Microsoft cloud revenue", index, sample_chunks,
                top_k=5, year_filter=2022,
            )

        for chunk in results:
            assert chunk["year"] == 2022

    def test_faiss_score_present(self, sample_chunks):
        from src.retrieval.retriever import faiss_search

        index, _ = self._make_index(n_vectors=len(sample_chunks), dim=8)

        with patch("src.retrieval.retriever.embed_query") as mock_embed:
            q_vec = np.random.rand(1, 8).astype("float32")
            import faiss as _faiss
            _faiss.normalize_L2(q_vec)
            mock_embed.return_value = q_vec
            results = faiss_search("Azure revenue growth", index, sample_chunks, top_k=2)

        for chunk in results:
            assert "faiss_score" in chunk
            assert isinstance(chunk["faiss_score"], float)


# ── API endpoints ─────────────────────────────────────────────────────────────

class TestAPI:
    @pytest.fixture()
    def client(self, sample_chunks):
        import src.api.api as api_module
        from fastapi.testclient import TestClient

        mock_pipeline = MagicMock()
        mock_pipeline.metadata = sample_chunks
        mock_pipeline.ask.return_value = {
            "query":       "What are Tesla risk factors?",
            "answer":      "Tesla faces macroeconomic risks [Tesla | pages 10–11].",
            "sources":     "**Sources used:**\n  • Tesla 10-K | pages 10–11",
            "chunks_used": sample_chunks[:1],
            "blocked":     False,
        }

        original = api_module.pipeline
        api_module.pipeline = mock_pipeline
        try:
            test_client = TestClient(api_module.app, raise_server_exceptions=False)
            yield test_client, mock_pipeline
        finally:
            api_module.pipeline = original

    def test_health_endpoint(self, client):
        test_client, _ = client
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ok", "starting")

    def test_companies_endpoint(self, client):
        test_client, _ = client
        response = test_client.get("/companies")
        assert response.status_code == 200
        companies = response.json()["companies"]
        assert set(companies) == {"tesla", "apple", "microsoft"}

    def test_ask_endpoint_success(self, client):
        test_client, _ = client
        response = test_client.post(
            "/ask",
            json={"query": "What are Tesla main risk factors?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data

    def test_ask_with_year_filter(self, client):
        test_client, mock_pipeline = client
        test_client.post(
            "/ask",
            json={"query": "Tesla risks", "year_filter": 2024},
        )
        call_kwargs = mock_pipeline.ask.call_args
        assert call_kwargs is not None

    def test_ask_invalid_company_filter(self, client):
        test_client, _ = client
        response = test_client.post(
            "/ask",
            json={"query": "What are risks?", "company_filter": "invalid_co"},
        )
        assert response.status_code == 422

    def test_ask_query_too_short(self, client):
        test_client, _ = client
        response = test_client.post("/ask", json={"query": "hi"})
        assert response.status_code == 422