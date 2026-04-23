"""基于 wfdb 的数据加载模块。

主要职责：
    - 从本地磁盘读取 MIT-BIH / PTB-XL 的 PhysioNet 标准文件 (.dat/.hea/.atr)；
    - 根据 .atr 标注中的 R 峰位置切分心拍；
    - 将标注符号映射为少数几类供分类任务使用；
    - 向外提供可用于 PyTorch DataLoader 的 `BeatDataset`。

严格遵循项目要求：**不做任何在线下载**。若本地路径不存在直接抛错。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import wfdb
from torch.utils.data import Dataset

from ..utils.logging_utils import get_logger
from .preprocess import apply_normalize, bandpass_filter, segment_by_rpeak

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# MIT-BIH
# ---------------------------------------------------------------------

def scan_mitbih_records(mitbih_dir: str | Path) -> List[str]:
    """扫描目录下所有 MIT-BIH 记录名（即去掉 .hea 后的文件名前缀）。"""
    mitbih_dir = Path(mitbih_dir)
    if not mitbih_dir.exists():
        raise FileNotFoundError(
            f"MIT-BIH directory not found: {mitbih_dir}. "
            "请从 PhysioNet 下载数据集并在 config.yaml 中配置路径。"
        )
    records = sorted(p.stem for p in mitbih_dir.glob("*.hea"))
    if not records:
        raise FileNotFoundError(
            f"No .hea files found under {mitbih_dir}. 请确认数据已正确解压。"
        )
    return records


def load_mitbih_record(
    record_path: str | Path,
    channel: int = 0,
) -> Tuple[np.ndarray, float, np.ndarray, List[str]]:
    """读取一条 MIT-BIH 记录。

    Args:
        record_path: 去掉扩展名的记录路径，例如 "/data/mitbih/100"。
        channel: 选择第几个通道（MIT-BIH 一般为 0=MLII, 1=V5）。

    Returns:
        signal: 一维 float32 数组。
        fs: 采样率。
        rpeaks: R 峰位置（样本下标）。
        symbols: 每个 R 峰对应的注释符号字符串列表。
    """
    record_path = str(record_path)
    record = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, extension="atr")

    sig = np.asarray(record.p_signal, dtype=np.float32)
    if sig.ndim != 2:
        raise ValueError(f"expect 2D signal from wfdb, got shape {sig.shape}")
    if channel >= sig.shape[1]:
        raise ValueError(
            f"channel {channel} out of range for record with {sig.shape[1]} channels"
        )
    signal = sig[:, channel]
    fs = float(record.fs)
    rpeaks = np.asarray(ann.sample, dtype=np.int64)
    symbols = list(ann.symbol)
    return signal, fs, rpeaks, symbols


def _map_symbol(symbol: str, symbol_map: Dict[str, str]) -> Optional[str]:
    """把 wfdb 注释符号映射为分类标签；未在表中的返回 None（将被丢弃）。"""
    return symbol_map.get(symbol)


def build_mitbih_beats(
    mitbih_dir: str | Path,
    classes: List[str],
    symbol_map: Dict[str, str],
    sampling_rate: float,
    bandpass_low: float,
    bandpass_high: float,
    filter_order: int,
    normalize: str,
    win_before: int,
    win_after: int,
    channel: int = 0,
    records: Optional[Iterable[str]] = None,
    max_records: int = 0,
    max_samples: int = 0,
    shuffle_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """遍历 MIT-BIH 目录，对每条记录完成 "滤波 → 归一化 → R 峰切分 → 标签映射"。

    若设置了 `max_samples`，在最终 shuffle 一次后再截断，避免前几条记录
    的类分布主导结果。

    Returns:
        X: 形状 `(N, win)` 的 float32 数组。
        y: 形状 `(N,)` 的 int64 标签数组。
        record_ids: 每个样本来自的记录名。
    """
    mitbih_dir = Path(mitbih_dir)
    all_records = list(records) if records else scan_mitbih_records(mitbih_dir)
    if max_records and max_records > 0:
        all_records = all_records[:max_records]

    class_to_idx = {c: i for i, c in enumerate(classes)}

    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    rec_parts: List[str] = []

    for rec in all_records:
        rec_path = mitbih_dir / rec
        signal, fs, rpeaks, symbols = load_mitbih_record(rec_path, channel=channel)
        fs_used = fs if abs(fs - sampling_rate) > 1e-3 else sampling_rate
        sig_f = bandpass_filter(
            signal, fs=fs_used, low=bandpass_low, high=bandpass_high, order=filter_order
        )
        sig_n = apply_normalize(sig_f, normalize)
        segs, kept_idx = segment_by_rpeak(sig_n, rpeaks, win_before, win_after)
        if segs.shape[0] == 0:
            continue
        kept_labels: List[int] = []
        kept_mask: List[int] = []
        for local_i, orig_i in enumerate(kept_idx):
            mapped = _map_symbol(symbols[orig_i], symbol_map)
            if mapped is None or mapped not in class_to_idx:
                continue
            kept_labels.append(class_to_idx[mapped])
            kept_mask.append(local_i)
        if not kept_labels:
            continue
        X_parts.append(segs[kept_mask])
        y_parts.append(np.asarray(kept_labels, dtype=np.int64))
        rec_parts.extend([rec] * len(kept_labels))
        logger.debug("record %s: %d beats after mapping", rec, len(kept_labels))

    if not X_parts:
        raise RuntimeError(
            "没有成功构造任何心拍样本。请检查 MIT-BIH 路径、标签映射与切分窗口配置。"
        )

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    record_ids = np.asarray(rec_parts)

    # 最终 shuffle + 截断，避免前几条记录主导类分布
    rng = np.random.default_rng(shuffle_seed)
    order = rng.permutation(X.shape[0])
    X, y, record_ids = X[order], y[order], record_ids[order]

    if max_samples and X.shape[0] > max_samples:
        X = X[:max_samples]
        y = y[:max_samples]
        record_ids = record_ids[:max_samples]
    return X, y, record_ids.tolist()


# ---------------------------------------------------------------------
# PTB-XL
# ---------------------------------------------------------------------

def load_ptbxl_index(csv_path: str | Path) -> pd.DataFrame:
    """读取 PTB-XL 索引 CSV，返回 DataFrame。

    **说明**：PTB-XL 使用 `ptbxl_database.csv` 提供记录级诊断标签
    (`scp_codes`)，**没有 .atr 标注文件**，因此不能直接用 MIT-BIH
    的 R 峰切分逻辑训练。本函数目前仅提供索引读取，完整训练链路
    需要在此基础上额外解析 `scp_codes` + 读取 `filename_lr/hr` 指定
    的 .dat/.hea，然后复用 `preprocess` 里的滤波/归一化函数按固定时
    长切分样本。作为多模态/记录级扩展的接口留存。
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"PTB-XL index CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


# ---------------------------------------------------------------------
# PyTorch Dataset 封装
# ---------------------------------------------------------------------

class BeatDataset(Dataset):
    """把 (X, y) 包装为 PyTorch Dataset。

    约定：
        - 输入 X 形状必须是 `(N, win)` 或 `(N, C, win)`；
        - 参数 `channels` 必须与 X 的通道数一致，否则抛错（不做隐式广播）。
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, channels: int = 1):
        if X.ndim == 2:
            X = X[:, None, :]
        if X.ndim != 3:
            raise ValueError(f"X must be (N, win) or (N, C, win), got shape {X.shape}")
        if X.shape[1] != channels:
            raise ValueError(
                f"BeatDataset channel mismatch: X has {X.shape[1]} channels but "
                f"model expects {channels}. 请确认 config.model.in_channels 与"
                " 数据准备逻辑一致（单导联 ECG 为 1）。"
            )
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"X/y length mismatch: {X.shape[0]} vs {y.shape[0]}")
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
