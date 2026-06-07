#!/usr/bin/env python3
"""阿里云千问 API 连通性测试。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    env = os.environ.copy()
    env["LLM_PROVIDER"] = "qwen"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "test_llm.py")],
        env=env,
        cwd=str(ROOT),
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
