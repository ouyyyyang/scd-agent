"""命令行推理入口。

用法：
    uv run python scripts/infer.py --config config/config.yaml --record 100
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scd_agent.cli import infer_entry  # noqa: E402


if __name__ == "__main__":
    sys.exit(infer_entry())
