"""命令行训练入口。

用法：
    uv run python scripts/train.py --config config/config.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

# 允许从仓库根目录直接 `python scripts/train.py` 调用
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scd_agent.cli import train_entry  # noqa: E402


if __name__ == "__main__":
    sys.exit(train_entry())
