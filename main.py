"""端到端演示入口：训练 → 保存 → 加载 → 推理 → 风险评估 → Agent 决策。

用法：
    uv run python main.py --config config/config.yaml
    uv run python main.py --smoke              # 小规模快速跑通
    uv run python main.py --skip-train --record 100

需要先在 `config/config.yaml` 中填入本地 MIT-BIH 数据路径，
或用 `--mitbih-dir` 从命令行覆盖。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scd_agent.data.loader import scan_mitbih_records  # noqa: E402
from scd_agent.pipeline import apply_smoke_preset, infer_pipeline, train_pipeline  # noqa: E402
from scd_agent.utils.config import ensure_dataset_paths, load_config  # noqa: E402
from scd_agent.utils.logging_utils import get_logger, setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="SCD Agent end-to-end demo")
    parser.add_argument("--config", "-c", default="config/config.yaml")
    parser.add_argument("--mitbih-dir", default=None)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--skip-train", action="store_true", help="跳过训练（需要已有 checkpoint）")
    parser.add_argument("--smoke", action="store_true", help="小规模快速跑通完整流程")
    parser.add_argument("--record", default=None, help="推理使用的记录名，默认取目录第一条")
    parser.add_argument("--max-beats", type=int, default=10)
    parser.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(args.log_level)
    log = get_logger("main")

    cfg = load_config(args.config)
    if args.mitbih_dir:
        cfg.raw.setdefault("data", {})["mitbih_dir"] = args.mitbih_dir
    if args.device:
        cfg.raw.setdefault("training", {})["device"] = args.device
    if args.checkpoint:
        cfg.raw.setdefault("training", {})["checkpoint_path"] = args.checkpoint
    if args.smoke:
        apply_smoke_preset(cfg)

    try:
        ensure_dataset_paths(cfg)
    except (RuntimeError, FileNotFoundError) as e:
        log.error("数据路径校验失败: %s", e)
        return 2

    if not args.skip_train:
        train_pipeline(cfg)

    record_name = args.record
    if record_name is None:
        records = scan_mitbih_records(cfg.data.mitbih_dir)
        record_name = records[0]
    record_path = Path(cfg.data.mitbih_dir) / record_name

    log.info("推理记录: %s", record_path)
    decisions = infer_pipeline(cfg, record_path, max_beats=args.max_beats)
    log.info("Agent 决策结果:")
    print(json.dumps(decisions, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
