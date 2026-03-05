"""
test_ingestion.py
-----------------
Unit tests for the ingestion pipeline (downloader + processor).

Run with:
    pytest tests/test_ingestion.py -v
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_TEXT = (
    "Tesla Inc. faces significant risks related to macroeconomic conditions. "
    "Interest rate fluctuations and inflation may adversely affect consumer "
    "demand for electric vehicles. The company operates in a highly competitive "
    "market and relies on the continued growth of the EV industry. "
    "Supply chain disruptions could materially impact production capacity. "
    "Regulatory changes in key markets such as China and Europe represent "
    "additional sources of uncertainty for the business. "
) * 20   # ~140 words × 20 repetitions = ~2800 words → several chunks


@pytest.fixture()
def tmp_dirs(tmp_path):
    """Create temporary raw and processed directories."""
    raw  = tmp_path / "raw"
    proc = tmp_path / "processed"
    raw.mkdir()
    proc.mkdir()
    return raw, proc


# ── clean_text ────────────────────────────────────────────────────────────────

class TestCleanText:
    def test_removes_control_characters(self):
        from src.ingestion.processor import clean_text
        dirty = "Hello\x00World\x07"
        result = clean_text(dirty)
        assert "\x00" not in result
        assert "\x07" not in result
        assert "Hello" in result
        assert "World" in result

    def test_collapses_excessive_spaces(self):
        from src.ingestion.processor import clean_text
        result = clean_text("word1     word2     word3")
        import re
        assert not re.search(r" {3,}", result)

    def test_removes_standalone_page_numbers(self):
        from src.ingestion.processor import clean_text
        text = "Introduction\n\n  42  \n\nConclusion"
        result = clean_text(text)
        lines = [line.strip() for line in result.splitlines() if line.strip()]
        assert "42" not in lines

    def test_preserves_financial_numbers(self):
        from src.ingestion.processor import clean_text
        text = "Revenue was $42.5 billion in fiscal year 2024."
        result = clean_text(text)
        assert "42.5" in result
        assert "billion" in result

    def test_empty_string(self):
        from src.ingestion.processor import clean_text
        assert clean_text("") == ""


# ── extract_year_from_filename ────────────────────────────────────────────────
# NOTE: the function expects a Path object, not a plain string.

class TestExtractYearFromFilename:
    def test_extracts_year_from_tesla(self):
        from src.ingestion.processor import extract_year_from_filename
        assert extract_year_from_filename(Path("tesla_10k_2024.htm")) == 2024

    def test_extracts_year_from_apple(self):
        from src.ingestion.processor import extract_year_from_filename
        assert extract_year_from_filename(Path("apple_10k_2022.htm")) == 2022

    def test_extracts_year_from_microsoft(self):
        from src.ingestion.processor import extract_year_from_filename
        assert extract_year_from_filename(Path("microsoft_10k_2023.html")) == 2023

    def test_returns_zero_for_legacy_filename(self):
        from src.ingestion.processor import extract_year_from_filename
        # Old-style filenames without year
        result = extract_year_from_filename(Path("tesla_10k.htm"))
        assert result == 0

    def test_returns_zero_for_unknown_format(self):
        from src.ingestion.processor import extract_year_from_filename
        result = extract_year_from_filename(Path("random_file.htm"))
        assert result == 0


# ── chunk_pages ───────────────────────────────────────────────────────────────

class TestChunkPages:
    def _make_pages(self, text: str, n_pages: int = 3) -> list[dict]:
        """Split text into equal-sized fake pages."""
        size = max(1, len(text) // n_pages)
        pages = []
        for i in range(n_pages):
            pages.append({"page": i + 1, "text": text[i * size:(i + 1) * size]})
        return pages

    def test_returns_list_of_dicts(self):
        from src.ingestion.processor import chunk_pages
        pages = self._make_pages(SAMPLE_TEXT)
        chunks = chunk_pages(pages)
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert isinstance(chunks[0], dict)

    def test_chunk_has_required_keys(self):
        from src.ingestion.processor import chunk_pages
        pages = self._make_pages(SAMPLE_TEXT)
        chunk = chunk_pages(pages)[0]
        for key in ("chunk_id", "text", "start_page", "end_page", "word_count"):
            assert key in chunk, f"Missing key: {key}"

    def test_chunk_ids_are_sequential(self):
        from src.ingestion.processor import chunk_pages
        pages = self._make_pages(SAMPLE_TEXT)
        chunks = chunk_pages(pages)
        ids = [c["chunk_id"] for c in chunks]
        assert ids == list(range(len(chunks)))

    def test_word_count_within_bounds(self):
        from src.ingestion.processor import CHUNK_SIZE, MIN_CHUNK, chunk_pages
        pages = self._make_pages(SAMPLE_TEXT)
        for chunk in chunk_pages(pages):
            assert chunk["word_count"] >= MIN_CHUNK
            assert chunk["word_count"] <= CHUNK_SIZE + 5

    def test_page_numbers_are_valid(self):
        from src.ingestion.processor import chunk_pages
        pages = self._make_pages(SAMPLE_TEXT, n_pages=5)
        for chunk in chunk_pages(pages):
            assert chunk["start_page"] >= 1
            assert chunk["end_page"] >= chunk["start_page"]

    def test_empty_pages_returns_empty(self):
        from src.ingestion.processor import chunk_pages
        assert chunk_pages([]) == []

    def test_overlap_creates_more_chunks_than_no_overlap(self):
        from src.ingestion.processor import chunk_pages
        pages = self._make_pages(SAMPLE_TEXT)
        chunks_with_overlap    = chunk_pages(pages, chunk_size=300, overlap=50)
        chunks_without_overlap = chunk_pages(pages, chunk_size=300, overlap=0)
        assert len(chunks_with_overlap) >= len(chunks_without_overlap)


# ── extract_from_html ─────────────────────────────────────────────────────────

class TestExtractFromHTML:
    def _write_html(self, tmp_path: Path, body: str) -> Path:
        html = f"<html><body>{body}</body></html>"
        path = tmp_path / "test.htm"
        path.write_text(html, encoding="utf-8")
        return path

    def test_extracts_paragraph_text(self, tmp_path):
        from src.ingestion.processor import extract_from_html
        path = self._write_html(
            tmp_path,
            "<p>" + ("Tesla faces regulatory risk. " * 5) + "</p>"
        )
        pages = extract_from_html(path)
        combined = " ".join(p["text"] for p in pages)
        assert "Tesla" in combined
        assert "regulatory" in combined

    def test_removes_script_tags(self, tmp_path):
        from src.ingestion.processor import extract_from_html
        path = self._write_html(
            tmp_path,
            "<script>alert('xss')</script><p>" + ("Real content here. " * 5) + "</p>"
        )
        pages = extract_from_html(path)
        combined = " ".join(p["text"] for p in pages)
        assert "alert" not in combined
        assert "Real content" in combined

    def test_page_numbers_start_at_one(self, tmp_path):
        from src.ingestion.processor import extract_from_html
        body = "<p>" + ("Financial data and analysis. " * 10) + "</p>"
        path = self._write_html(tmp_path, body)
        pages = extract_from_html(path)
        if pages:
            assert pages[0]["page"] == 1


# ── process_document ─────────────────────────────────────────────────────────
# NOTE: process_document returns chunks but does NOT save JSON itself.
# JSON saving happens in main() after aggregating all companies.

class TestProcessDocument:
    def test_returns_list_of_chunks(self, tmp_dirs, tmp_path):
        from src.ingestion.processor import process_document
        raw_dir, _ = tmp_dirs

        html_content = "<html><body>" + \
                       "<p>" + ("Apple Inc. reports strong Services revenue. " * 30) + "</p>" + \
                       "</body></html>"
        htm_file = raw_dir / "apple_10k_2024.htm"
        htm_file.write_text(html_content, encoding="utf-8")

        chunks = process_document(htm_file, "apple")

        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_chunks_have_company_and_year_metadata(self, tmp_dirs):
        from src.ingestion.processor import process_document
        raw_dir, _ = tmp_dirs

        html_content = "<html><body>" + \
                       "<p>" + ("Microsoft Azure cloud computing segment. " * 30) + "</p>" + \
                       "</body></html>"
        htm_file = raw_dir / "microsoft_10k_2023.htm"
        htm_file.write_text(html_content, encoding="utf-8")

        chunks = process_document(htm_file, "microsoft")

        for chunk in chunks:
            assert chunk["company"] == "microsoft"
            assert chunk["source"]  == "microsoft_10k_2023.htm"
            assert chunk["year"]    == 2023   # extracted from filename

    def test_chunks_year_zero_for_legacy_filename(self, tmp_dirs):
        """Legacy filenames without year should produce year=0."""
        from src.ingestion.processor import process_document
        raw_dir, _ = tmp_dirs

        html_content = "<html><body>" + \
                       "<p>" + ("Tesla motors electric vehicle. " * 30) + "</p>" + \
                       "</body></html>"
        htm_file = raw_dir / "tesla_10k.htm"
        htm_file.write_text(html_content, encoding="utf-8")

        chunks = process_document(htm_file, "tesla")

        for chunk in chunks:
            assert chunk["year"] == 0

    def test_multi_year_chunks_have_correct_years(self, tmp_dirs):
        """Processing two files for the same company yields correct years on each chunk."""
        from src.ingestion.processor import process_document
        raw_dir, _ = tmp_dirs

        body = "<p>" + ("Tesla risk factors for the period. " * 30) + "</p>"
        html = f"<html><body>{body}</body></html>"

        file_2022 = raw_dir / "tesla_10k_2022.htm"
        file_2024 = raw_dir / "tesla_10k_2024.htm"
        file_2022.write_text(html, encoding="utf-8")
        file_2024.write_text(html, encoding="utf-8")

        chunks_2022 = process_document(file_2022, "tesla")
        chunks_2024 = process_document(file_2024, "tesla")

        assert all(c["year"] == 2022 for c in chunks_2022)
        assert all(c["year"] == 2024 for c in chunks_2024)


# ── downloader ────────────────────────────────────────────────────────────────

class TestDownloader:
    def test_get_recent_10k_filings_returns_list(self):
        """get_recent_10k_filings parses EDGAR API response correctly."""
        from src.ingestion.downloader import get_recent_10k_filings

        mock_data = {
            "filings": {
                "recent": {
                    "form":            ["8-K", "10-Q", "10-K", "10-K"],
                    "accessionNumber": ["0001-00", "0002-00", "0003-00", "0004-00"],
                    "primaryDocument": ["8k.htm", "10q.htm", "10k_2024.htm", "10k_2023.htm"],
                    "filingDate":      ["2024-01-01", "2024-04-01", "2024-10-31", "2023-10-31"],
                }
            }
        }

        with patch("src.ingestion.downloader.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_data
            mock_get.return_value = mock_response

            results = get_recent_10k_filings("0001318605", n=2)

        assert isinstance(results, list)
        assert len(results) == 2
        for filing in results:
            assert "accession"  in filing
            assert "primary"    in filing
            assert "year"       in filing
            assert "filed_date" in filing

    def test_get_recent_10k_filings_deduplicates_years(self):
        """Should return at most one filing per fiscal year."""
        from src.ingestion.downloader import get_recent_10k_filings

        mock_data = {
            "filings": {
                "recent": {
                    "form":            ["10-K", "10-K/A", "10-K"],
                    "accessionNumber": ["0001-00", "0002-00", "0003-00"],
                    "primaryDocument": ["10k_2024.htm", "10k_2024a.htm", "10k_2023.htm"],
                    "filingDate":      ["2024-10-31", "2024-11-15", "2023-10-31"],
                }
            }
        }

        with patch("src.ingestion.downloader.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_data
            mock_get.return_value = mock_response

            results = get_recent_10k_filings("0001318605", n=4)

        years = [r["year"] for r in results]
        assert len(years) == len(set(years))   # no duplicate years

    def test_get_recent_10k_filings_respects_n_limit(self):
        """Should return at most n filings."""
        from src.ingestion.downloader import get_recent_10k_filings

        mock_data = {
            "filings": {
                "recent": {
                    "form":            ["10-K", "10-K", "10-K", "10-K"],
                    "accessionNumber": ["0001-00", "0002-00", "0003-00", "0004-00"],
                    "primaryDocument": ["a.htm", "b.htm", "c.htm", "d.htm"],
                    "filingDate":      ["2025-10-31", "2024-10-31", "2023-10-31", "2022-10-31"],
                }
            }
        }

        with patch("src.ingestion.downloader.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_data
            mock_get.return_value = mock_response

            results = get_recent_10k_filings("0001318605", n=2)

        assert len(results) <= 2

    def test_build_document_url_format(self):
        from src.ingestion.downloader import build_document_url

        filing = {
            "cik_short": "1318605",
            "accession":  "000131860524000010",
            "primary":    "tsla-20231231.htm",
        }
        url = build_document_url(filing)

        assert url.startswith("https://www.sec.gov/Archives/edgar/data/")
        assert "1318605" in url
        assert "tsla-20231231.htm" in url
