"""控制台脚本入口，由 pyproject.toml 注册为命令：
    scd-agent-train
    scd-agent-infer

提供 CLI 覆盖参数：`--mitbih-dir / --epochs / --batch-size / --device /
--max-samples / --checkpoint / --smoke / --log-level`，方便不改 YAML 就调整运行参数。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from .pipeline import apply_smoke_preset, infer_pipeline, train_pipeline
from .utils.config import Config, load_config, resolve_project_path
from .utils.logging_utils import setup_logging


# ---------------------------------------------------------------------
# 公共参数
# ---------------------------------------------------------------------

def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", "-c", default="config/config.yaml", help="YAML 配置文件路径")
    p.add_argument("--mitbih-dir", default=None, help="覆盖 data.mitbih_dir")
    p.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"], help="覆盖 training.device")
    p.add_argument("--checkpoint", default=None, help="覆盖 training.checkpoint_path")
    p.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别，默认 INFO",
    )
    p.add_argument("--smoke", action="store_true", help="使用小规模预设（1 epoch / <=3 记录 / <=1000 样本）")


def _apply_overrides(cfg: Config, args: argparse.Namespace, *, training_overrides: Dict[str, Any] | None = None) -> Config:
    if args.mitbih_dir:
        cfg.raw.setdefault("data", {})["mitbih_dir"] = args.mitbih_dir
    if args.device:
        cfg.raw.setdefault("training", {})["device"] = args.device
    if args.checkpoint:
        cfg.raw.setdefault("training", {})["checkpoint_path"] = args.checkpoint
    if training_overrides:
        cfg.raw.setdefault("training", {}).update({k: v for k, v in training_overrides.items() if v is not None})
    if args.smoke:
        apply_smoke_preset(cfg)
    return cfg


# ---------------------------------------------------------------------
# train
# ---------------------------------------------------------------------

def train_entry(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="训练 CNN-LSTM 心律失常分类模型")
    _add_common_args(parser)
    parser.add_argument("--epochs", type=int, default=None, help="覆盖 training.epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖 training.batch_size")
    parser.add_argument("--lr", type=float, default=None, help="覆盖 training.learning_rate")
    parser.add_argument("--max-samples", type=int, default=None, help="覆盖 training.max_samples")
    parser.add_argument("--class-weight", default=None, choices=["inverse_freq", "none"], help="覆盖 training.class_weight")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    cfg = load_config(args.config)
    _apply_overrides(
        cfg,
        args,
        training_overrides={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "max_samples": args.max_samples,
            "class_weight": args.class_weight,
        },
    )
    try:
        train_pipeline(cfg)
    except (RuntimeError, FileNotFoundError) as e:
        from .utils.logging_utils import get_logger
        get_logger("cli").error("训练失败: %s", e)
        return 2
    return 0


# ---------------------------------------------------------------------
# infer
# ---------------------------------------------------------------------

def infer_entry(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="在单条 MIT-BIH 记录上推理 + 风险评估 + Agent 决策")
    _add_common_args(parser)
    parser.add_argument("--record", required=True, help="MIT-BIH 记录名，如 100")
    parser.add_argument("--max-beats", type=int, default=20, help="最多输出多少个心拍的决策")
    parser.add_argument("--output", default=None, help="将结构化结果写入 JSON 文件")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    cfg = load_config(args.config)
    _apply_overrides(cfg, args)

    record_path = Path(cfg.data.mitbih_dir) / args.record
    try:
        decisions = infer_pipeline(cfg, record_path, max_beats=args.max_beats)
    except (RuntimeError, FileNotFoundError) as e:
        from .utils.logging_utils import get_logger
        get_logger("cli").error("推理失败: %s", e)
        return 2
    payload = json.dumps(decisions, ensure_ascii=False, indent=2)
    if args.output:
        out = resolve_project_path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
        print(f"已写入 {out}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(train_entry())
