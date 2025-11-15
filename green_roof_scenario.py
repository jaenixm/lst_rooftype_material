#!/usr/bin/env python3
"""Backward-compatible CLI entry point for the green_roof_scenario package."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent
_SRC_PATH = _REPO_ROOT / "src"
if _SRC_PATH.exists() and str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from green_roof_scenario.cli import main


if __name__ == "__main__":
    main()
