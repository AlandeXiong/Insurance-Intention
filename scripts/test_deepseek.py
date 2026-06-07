#!/usr/bin/env python3
"""DeepSeek API connectivity test (legacy entry point; prefer scripts/test_llm.py)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("LLM_PROVIDER", "deepseek")

from scripts.test_llm import main

if __name__ == "__main__":
    main()
