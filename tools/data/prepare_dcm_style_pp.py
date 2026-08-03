"""CLI wrapper for preparing the portable DCM-style++ NPY dataset."""

# ruff: noqa: E402

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.prepare import main


if __name__ == "__main__":
    main()
