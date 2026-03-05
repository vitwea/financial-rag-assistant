"""
processor.py
------------
Converts raw 10-K filings (HTML or PDF) into clean, overlapping text chunks
with metadata including company name AND fiscal year.

File naming convention expected from downloader:
    data/raw/{company}_10k_{year}{ext}
    e.g. tesla_10k_2024.htm, apple_10k_2025.htm

Each chunk JSON record now includes:
    company, year, source, chunk_id, start_page, end_page, word_count, text

Usage:
    python -m src.ingestion.processor
"""

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

CHUNK_SIZE    = 500
OVERLAP       = 50
MIN_CHUNK     = 50
MIN_BLOCK_LEN = 80


# ── Text extraction ───────────────────────────────────────────────────────────

def extract_from_html(file_path: Path) -> list[dict]:
    raw  = file_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")

    for tag in soup(["script", "style", "head", "meta", "link", "ix:header"]):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        tag.decompose()

    content_tags = ["p", "div", "section", "span", "td", "li",
                    "h1", "h2", "h3", "h4", "h5"]
    blocks = []
    for element in soup.find_all(content_tags):
        if element.find(content_tags):
            continue
        text = element.get_text(separator=" ", strip=True)
        if len(text) >= MIN_BLOCK_LEN:
            blocks.append(text)

    combined      = " ".join(blocks)
    chars_per_page = 3_000
    pages = []
    for idx, start in enumerate(range(0, len(combined), chars_per_page)):
        pages.append({"page": idx + 1, "text": combined[start: start + chars_per_page]})

    logger.debug("Extracted %d virtual pages from HTML (%d chars)", len(pages), len(combined))
    return pages


def extract_from_pdf(file_path: Path) -> list[dict]:
    reader = PdfReader(file_path)
    pages  = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    logger.debug("Extracted %d pages from PDF", len(pages))
    return pages


def extract_text(file_path: Path) -> list[dict]:
    suffix = file_path.suffix.lower()
    if suffix in {".htm", ".html"}:
        return extract_from_html(file_path)
    if suffix == ".pdf":
        return extract_from_pdf(file_path)
    raise ValueError(f"Unsupported file type: {suffix}")


# ── Cleaning ──────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r" {3,}", "  ", text)
    text = re.sub(r"\n{4,}", "\n\n", text)
    text = re.sub(r"(?m)^\s*\d{1,3}\s*$", "", text)
    text = re.sub(r"(?i)table of contents", "", text)
    return text.strip()


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_pages(pages: list[dict],
                chunk_size: int = CHUNK_SIZE,
                overlap: int = OVERLAP) -> list[dict]:
    word_list: list[tuple[str, int]] = []
    for page in pages:
        cleaned = clean_text(page["text"])
        for word in cleaned.split():
            word_list.append((word, page["page"]))

    chunks   = []
    chunk_id = 0
    i        = 0
    step     = chunk_size - overlap

    while i < len(word_list):
        window = word_list[i: i + chunk_size]
        if len(window) < MIN_CHUNK:
            break

        words  = [w for w, _ in window]
        pages_ = [p for _, p in window]

        chunks.append({
            "chunk_id":   chunk_id,
            "text":       " ".join(words),
            "start_page": pages_[0],
            "end_page":   pages_[-1],
            "word_count": len(words),
        })
        chunk_id += 1
        i += step

    logger.debug("Created %d chunks from %d pages", len(chunks), len(pages))
    return chunks


# ── Year extraction ───────────────────────────────────────────────────────────

def extract_year_from_filename(file_path: Path) -> int:
    """
    Extract fiscal year from filename convention: {company}_10k_{year}.ext
    Falls back to 0 if the year cannot be parsed.
    """
    stem  = file_path.stem           # e.g. "tesla_10k_2024"
    parts = stem.split("_")
    for part in reversed(parts):
        if part.isdigit() and len(part) == 4:
            return int(part)
    logger.warning("Could not extract year from filename: %s", file_path.name)
    return 0


# ── Document pipeline ─────────────────────────────────────────────────────────

def process_document(file_path: Path, company: str) -> list[dict]:
    """Full pipeline for a single 10-K file: extract → chunk → annotate."""
    logger.info("Processing: %s", file_path.name)

    year   = extract_year_from_filename(file_path)
    pages  = extract_text(file_path)
    chunks = chunk_pages(pages)

    for chunk in chunks:
        chunk["company"] = company
        chunk["year"]    = year          # ← NEW
        chunk["source"]  = file_path.name

    logger.info("  %s %d → %d chunks", company.upper(), year, len(chunks))
    return chunks


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Find all raw 10-K files matching {company}_10k_{year}.ext
    pattern = "*_10k_*.htm *_10k_*.html *_10k_*.pdf".split()
    all_files: list[Path] = []
    for pat in pattern:
        all_files.extend(RAW_DIR.glob(pat))

    # Also support old single-year naming: {company}_10k.htm
    for company in ["tesla", "apple", "microsoft"]:
        for ext in [".htm", ".html", ".pdf"]:
            old_path = RAW_DIR / f"{company}_10k{ext}"
            if old_path.exists() and old_path not in all_files:
                all_files.append(old_path)

    if not all_files:
        logger.error("No raw 10-K files found in %s. Run downloader.py first.", RAW_DIR)
        return

    # Group chunks by company — save one JSON per company with ALL years
    company_chunks: dict[str, list[dict]] = {}
    companies = ["tesla", "apple", "microsoft"]

    for file_path in sorted(all_files):
        # Determine company from filename
        company = None
        for c in companies:
            if file_path.name.startswith(c):
                company = c
                break
        if company is None:
            logger.warning("Unknown company for file: %s — skipping", file_path.name)
            continue

        try:
            chunks = process_document(file_path, company)
            if company not in company_chunks:
                company_chunks[company] = []
            company_chunks[company].extend(chunks)
        except Exception as exc:
            logger.error("Failed to process %s: %s", file_path.name, exc)

    # Re-index chunk_ids globally per company and save
    for company, chunks in company_chunks.items():
        for i, chunk in enumerate(chunks):
            chunk["chunk_id"] = i

        output_path = PROCESSED_DIR / f"{company}_chunks.json"
        output_path.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        years = sorted({c["year"] for c in chunks})
        logger.info(
            "Saved %s → %d chunks across years %s",
            output_path.name, len(chunks), years,
        )

    # Summary
    logger.info("─── Summary ───────────────────────────────")
    total = 0
    for company, chunks in company_chunks.items():
        years  = sorted({c["year"] for c in chunks})
        logger.info("  %-12s %d chunks | years: %s", company, len(chunks), years)
        total += len(chunks)
    logger.info("  TOTAL: %d chunks", total)


if __name__ == "__main__":
    main()