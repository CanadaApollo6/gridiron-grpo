"""Make the flat `src/` layout importable in tests, independent of the pytest
`pythonpath` ini (belt and suspenders, and keeps `pytest` working from any cwd).
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
