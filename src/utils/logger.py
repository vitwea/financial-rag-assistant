"""
logger.py
---------
Centralised logging configuration for the Financial RAG Assistant.

All modules import from here instead of using print() or configuring
logging independently. This gives a consistent format across the entire
project and makes it trivial to change log level or output destination.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)

    logger.info("Index loaded: %d vectors", n)
    logger.warning("No chunks found for company: %s", company)
    logger.error("API call failed: %s", exc)
    logger.debug("Query expanded to: %s", expanded_query)
"""

import logging
import sys
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

LOG_DIR      = Path("logs")
LOG_FILE     = LOG_DIR / "rag_assistant.log"
LOG_LEVEL    = logging.DEBUG      # change to INFO in production
LOG_FORMAT   = "%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s"
DATE_FORMAT  = "%Y-%m-%d %H:%M:%S"


# ── Formatter ─────────────────────────────────────────────────────────────────

class _ColourFormatter(logging.Formatter):
    """
    Adds ANSI colour codes to log levels for terminal readability.
    Colours are stripped automatically when output is not a TTY
    (e.g. when piped to a file).
    """

    COLOURS = {
        logging.DEBUG:    "\033[37m",    # white
        logging.INFO:     "\033[36m",    # cyan
        logging.WARNING:  "\033[33m",    # yellow
        logging.ERROR:    "\033[31m",    # red
        logging.CRITICAL: "\033[1;31m",  # bold red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        if sys.stderr.isatty():
            colour = self.COLOURS.get(record.levelno, "")
            return f"{colour}{formatted}{self.RESET}"
        return formatted


# ── Root logger setup (runs once at import time) ──────────────────────────────

def _setup_root_logger() -> None:
    """
    Configure the root logger with:
      - A coloured StreamHandler for the terminal (DEBUG+)
      - A plain FileHandler for persistent logs (DEBUG+)

    Called once when this module is first imported.
    """
    root = logging.getLogger()

    # Avoid adding handlers multiple times if the module is re-imported
    if root.handlers:
        return

    root.setLevel(LOG_LEVEL)

    # ── Terminal handler ───────────────────────────────────────────────────
    # INFO and above only — DEBUG is too verbose for the console
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(
        _ColourFormatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
    )
    root.addHandler(stream_handler)

    # ── File handler ───────────────────────────────────────────────────────
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(
        logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
    )
    root.addHandler(file_handler)


_setup_root_logger()


# ── Public API ────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger inheriting from the configured root logger.

    Args:
        name : typically __name__ of the calling module

    Returns:
        logging.Logger instance ready to use

    Example:
        logger = get_logger(__name__)
        logger.info("Processing %d chunks", len(chunks))
    """
    return logging.getLogger(name)