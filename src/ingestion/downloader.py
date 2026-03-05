"""
downloader.py
-------------
Downloads the N most recent 10-K annual reports for Tesla, Apple, and
Microsoft using the SEC EDGAR submissions API.

Files are saved as:
    data/raw/{company}_10k_{year}{ext}
    e.g. data/raw/tesla_10k_2024.htm
         data/raw/apple_10k_2025.htm

Usage:
    python -m src.ingestion.downloader
"""

from pathlib import Path
import time

import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

RAW_DIR = Path("data/raw")

HEADERS = {
    "User-Agent": "YourName your@email.com",
    "Accept-Encoding": "gzip, deflate",
}

COMPANIES = {
    "tesla":     "0001318605",
    "apple":     "0000320193",
    "microsoft": "0000789019",
}

YEARS_TO_FETCH = 4
REQUEST_DELAY  = 0.6


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_recent_10k_filings(cik: str, n: int = YEARS_TO_FETCH) -> list[dict]:
    """
    Return metadata for the N most recent 10-K filings (one per fiscal year).
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()

    data   = response.json()
    recent = data["filings"]["recent"]

    filings: list[dict] = []
    seen_years: set[str] = set()

    for i, form_type in enumerate(recent["form"]):
        if form_type != "10-K":
            continue

        filed_date = recent["filingDate"][i]
        year       = filed_date[:4]

        if year in seen_years:
            continue
        seen_years.add(year)

        filings.append({
            "accession":  recent["accessionNumber"][i].replace("-", ""),
            "primary":    recent["primaryDocument"][i],
            "filed_date": filed_date,
            "year":       int(year),
            "cik_short":  str(int(cik)),
        })

        if len(filings) == n:
            break

    if not filings:
        raise ValueError(f"No 10-K filings found for CIK {cik}")

    return filings


def build_document_url(filing: dict) -> str:
    base = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{filing['cik_short']}/{filing['accession']}"
    )
    return f"{base}/{filing['primary']}"


def download_filing(company: str, filing: dict) -> Path:
    """Download a single 10-K filing. Skips if already exists."""
    ext         = Path(filing["primary"]).suffix
    output_path = RAW_DIR / f"{company}_10k_{filing['year']}{ext}"

    if output_path.exists():
        size_kb = output_path.stat().st_size / 1024
        logger.info(
            "Already downloaded %s %d (%.0f KB) — skipping",
            company.upper(), filing["year"], size_kb,
        )
        return output_path

    doc_url = build_document_url(filing)
    logger.info(
        "Downloading %s %d (filed %s)...",
        company.upper(), filing["year"], filing["filed_date"],
    )

    response = requests.get(doc_url, headers=HEADERS, timeout=120)
    response.raise_for_status()

    output_path.write_bytes(response.content)
    size_kb = output_path.stat().st_size / 1024
    logger.info("Saved: %s (%.0f KB)", output_path.name, size_kb)
    return output_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    total_ok, total_fail = 0, 0

    for company, cik in COMPANIES.items():
        logger.info("─── %s ───────────────────────────────", company.upper())
        try:
            filings = get_recent_10k_filings(cik, n=YEARS_TO_FETCH)
            logger.info("Found %d 10-K filings", len(filings))

            for filing in filings:
                try:
                    download_filing(company, filing)
                    total_ok += 1
                    time.sleep(REQUEST_DELAY)
                except Exception as exc:
                    logger.error("Failed %s %d: %s", company, filing["year"], exc)
                    total_fail += 1

        except Exception as exc:
            logger.error("Failed to list filings for %s: %s", company, exc)
            total_fail += 1

    logger.info("Download complete — OK: %d | Failed: %d", total_ok, total_fail)


if __name__ == "__main__":
    main()
