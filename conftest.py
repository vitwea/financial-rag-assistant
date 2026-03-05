# conftest.py
# -----------
# Shared pytest configuration. Runs automatically before any test file.

import sys
from pathlib import Path

# Ensure the project root is in sys.path so all `src.*` imports resolve
# regardless of where pytest is invoked from.
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
