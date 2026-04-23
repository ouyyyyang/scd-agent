"""统一的日志配置。

- `setup_logging(level)` 在入口脚本中调用一次；
- 各模块用 `get_logger(__name__)` 获取 logger。
"""
from __future__ import annotations

import logging
import os

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def setup_logging(level: str | int | None = None) -> None:
    """初始化根 logger。优先级：参数 > 环境变量 SCD_LOG_LEVEL > INFO。"""
    if level is None:
        level = os.environ.get("SCD_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format=_DEFAULT_FORMAT,
        datefmt=_DATE_FORMAT,
        force=True,
    )
    # wfdb / matplotlib 之类的三方日志只想看 WARNING 以上
    for noisy in ("matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
