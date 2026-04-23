"""端到端流水线工具。

把 "读取 → 预处理 → 训练 → 推理 → 风险评估 → Agent 决策" 这条路径上
的核心步骤封装成几个可复用函数，供 `main.py` 与 `scripts/` 下的
entry-point 脚本共用，避免重复代码。
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .agent.decision import RuleBasedAgent
from .data.loader import BeatDataset, build_mitbih_beats, load_mitbih_record
from .data.preprocess import apply_normalize, bandpass_filter, segment_by_rpeak
from .models.cnn_lstm import build_model
from .risk.assessor import RiskAssessor
from .training.trainer import (
    Trainer,
    classification_report_from_loader,
    compute_class_weights,
    load_checkpoint,
    predict_proba,
    resolve_device,
    save_checkpoint,
)
from .utils.config import Config, ensure_dataset_paths, resolve_project_path
from .utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# 通用
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_smoke_preset(cfg: Config) -> None:
    """把配置调成最小规模以便快速跑通全链路。"""
    cfg.raw.setdefault("training", {})
    cfg.raw["training"]["epochs"] = 1
    cfg.raw["training"]["max_samples"] = 1000
    cfg.raw["training"]["batch_size"] = min(int(cfg.raw["training"].get("batch_size", 64)), 64)
    cfg.raw.setdefault("data", {})
    cfg.raw["data"]["mitbih_max_records"] = min(
        int(cfg.raw["data"].get("mitbih_max_records", 10) or 10), 3
    )
    logger.info("smoke preset applied: epochs=1, max_samples=1000, max_records<=3")


# ---------------------------------------------------------------------
# 数据集构造
# ---------------------------------------------------------------------

def build_dataset(cfg: Config) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """根据配置读取 MIT-BIH 并构造 (X, y, record_ids)。"""
    data = cfg.data
    pre = cfg.preprocess
    labels = cfg.labels

    symbol_map = labels.symbol_map.to_dict() if hasattr(labels.symbol_map, "to_dict") else dict(labels.symbol_map)
    classes = list(labels.classes)

    records = list(data.mitbih_records) if data.get("mitbih_records") else None

    return build_mitbih_beats(
        mitbih_dir=data.mitbih_dir,
        classes=classes,
        symbol_map=symbol_map,
        sampling_rate=float(pre.sampling_rate),
        bandpass_low=float(pre.bandpass_low),
        bandpass_high=float(pre.bandpass_high),
        filter_order=int(pre.filter_order),
        normalize=str(pre.normalize),
        win_before=int(pre.win_before),
        win_after=int(pre.win_after),
        channel=int(data.channel),
        records=records,
        max_records=int(data.get("mitbih_max_records", 0) or 0),
        max_samples=int(cfg.training.get("max_samples", 0) or 0),
        shuffle_seed=int(cfg.training.seed),
    )


def make_loaders(
    X: np.ndarray,
    y: np.ndarray,
    cfg: Config,
) -> Tuple[DataLoader, DataLoader]:
    """按标签分层切分 train/val，返回两个 DataLoader。"""
    in_channels = int(cfg.model.in_channels)
    val_ratio = float(cfg.training.val_split)
    seed = int(cfg.training.seed)

    # 分层切分：保证 train/val 里类分布一致
    unique, counts = np.unique(y, return_counts=True)
    if (counts < 2).any():
        # 某类只有一条样本时 stratify 会报错，降级为普通切分
        logger.warning(
            "class has <2 samples (%s); fall back to non-stratified split",
            dict(zip(unique.tolist(), counts.tolist())),
        )
        stratify = None
    else:
        stratify = y
    idx_train, idx_val = train_test_split(
        np.arange(X.shape[0]),
        test_size=val_ratio,
        random_state=seed,
        stratify=stratify,
        shuffle=True,
    )
    train_ds = BeatDataset(X[idx_train], y[idx_train], channels=in_channels)
    val_ds = BeatDataset(X[idx_val], y[idx_val], channels=in_channels)

    bs = int(cfg.training.batch_size)
    nw = int(cfg.training.num_workers)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
    return train_loader, val_loader


# ---------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------

def train_pipeline(cfg: Config) -> Path:
    """执行一次完整训练并保存 checkpoint，返回保存路径。"""
    ensure_dataset_paths(cfg)
    set_seed(int(cfg.training.seed))

    logger.info("构造训练数据集 ...")
    X, y, _ = build_dataset(cfg)
    num_classes = int(cfg.model.num_classes)
    class_dist = np.bincount(y, minlength=num_classes).tolist()
    logger.info(
        "样本总数: %d，窗口长度: %d，类别分布 %s: %s",
        X.shape[0], X.shape[1], list(cfg.labels.classes), class_dist,
    )

    train_loader, val_loader = make_loaders(X, y, cfg)
    device = resolve_device(str(cfg.training.device))
    logger.info("使用设备: %s", device)

    # 类权重：仅对训练集计算，避免用 val 信息
    y_train = np.asarray([train_loader.dataset.y[i] for i in range(len(train_loader.dataset))])
    scheme = str(cfg.training.get("class_weight", "inverse_freq"))
    class_weights = compute_class_weights(y_train, num_classes=num_classes, scheme=scheme)
    logger.info("类权重 (%s): %s", scheme, class_weights.tolist())

    model = build_model(cfg)
    trainer = Trainer(
        model=model,
        device=device,
        lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
        class_weights=class_weights if scheme != "none" else None,
    )
    trainer.fit(train_loader, val_loader, epochs=int(cfg.training.epochs))

    # 分类报告：真正评估模型每类表现
    logger.info("生成验证集分类报告 ...")
    report = classification_report_from_loader(
        model, val_loader, device=device, class_names=list(cfg.labels.classes)
    )
    logger.info("\n%s", report)

    ckpt_path = resolve_project_path(cfg.training.checkpoint_path)
    meta = {
        "classes": list(cfg.labels.classes),
        "model": cfg.model.to_dict(),
        "preprocess": cfg.preprocess.to_dict(),
        "class_distribution": class_dist,
    }
    save_checkpoint(model, ckpt_path, meta=meta)
    logger.info("模型已保存到 %s", ckpt_path)
    return ckpt_path


# ---------------------------------------------------------------------
# 推理 + 风险 + 决策
# ---------------------------------------------------------------------

def preprocess_record_for_inference(
    cfg: Config, record_path: str | Path
) -> Tuple[np.ndarray, List[int]]:
    """读取单条记录并按训练相同的预处理流程切分心拍，返回 X 与 R 峰索引。"""
    pre = cfg.preprocess
    signal, fs, rpeaks, _ = load_mitbih_record(record_path, channel=int(cfg.data.channel))
    sig_f = bandpass_filter(
        signal,
        fs=fs,
        low=float(pre.bandpass_low),
        high=float(pre.bandpass_high),
        order=int(pre.filter_order),
    )
    sig_n = apply_normalize(sig_f, str(pre.normalize))
    X, kept = segment_by_rpeak(sig_n, rpeaks, int(pre.win_before), int(pre.win_after))
    return X, [int(rpeaks[i]) for i in kept]


def infer_pipeline(
    cfg: Config,
    record_path: str | Path,
    max_beats: int = 20,
) -> List[Dict[str, Any]]:
    """对单条记录执行推理 → 风险评估 → Agent 决策，返回结构化结果列表。"""
    ensure_dataset_paths(cfg)
    device = resolve_device(str(cfg.training.device))
    model = build_model(cfg)
    ckpt_path = resolve_project_path(cfg.training.checkpoint_path)
    meta = load_checkpoint(model, ckpt_path, map_location=str(device))
    model.to(device)

    # 优先使用 checkpoint 内记录的类顺序，保证训练/推理一致
    classes = list(meta.get("classes") or cfg.labels.classes)

    X, rpeak_samples = preprocess_record_for_inference(cfg, record_path)
    if X.shape[0] == 0:
        logger.warning("该记录未能切分出任何心拍")
        return []
    if max_beats and X.shape[0] > max_beats:
        X = X[:max_beats]
        rpeak_samples = rpeak_samples[:max_beats]

    probs = predict_proba(model, X, device=device)

    assessor = RiskAssessor.from_config(cfg, classes=classes)
    agent = RuleBasedAgent.from_config(cfg)

    decisions: List[Dict[str, Any]] = []
    record_name = Path(str(record_path)).name
    for i, p in enumerate(probs):
        result = assessor.assess(p)
        decision = agent.decide(
            result,
            meta={"record": record_name, "beat_index": i, "rpeak_sample": rpeak_samples[i]},
        )
        decisions.append(decision)
    return decisions
