#!/usr/bin/env python3
"""Stable command-line entry for the vector PDF extraction pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SOURCE_DIR))

from pipeline import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
