"""信号预处理函数集合。

包含：
    - Butterworth 带通滤波（推荐 0.5~40Hz 去除基线漂移与工频噪声）；
    - 归一化（Z-score / Min-Max）；
    - 基于 R 峰位置的固定窗口切分；
    - 基于固定时长的滑动切分（多模态扩展使用）。

所有函数都假设输入为形状为 `(n_samples,)` 或 `(n_samples, n_channels)`
的 `numpy.ndarray`。
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from scipy.signal import butter, filtfilt


# ---------------------------------------------------------------------
# 滤波
# ---------------------------------------------------------------------

def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low: float = 0.5,
    high: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """对一维或二维信号做零相位 Butterworth 带通滤波。

    Args:
        signal: 形状 `(n,)` 或 `(n, channels)`。
        fs: 采样率 (Hz)。
        low, high: 带通上下限 (Hz)。
        order: 滤波器阶数。
    """
    if signal.size == 0:
        return signal
    nyq = 0.5 * fs
    low_norm = max(low / nyq, 1e-6)
    high_norm = min(high / nyq, 0.999999)
    b, a = butter(order, [low_norm, high_norm], btype="band")
    # filtfilt 需要沿时间轴滤波
    axis = 0 if signal.ndim == 2 else -1
    return filtfilt(b, a, signal, axis=axis).astype(np.float32)


# ---------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------

def z_normalize(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """按样本做 Z-score 归一化。"""
    mean = signal.mean()
    std = signal.std()
    return ((signal - mean) / (std + eps)).astype(np.float32)


def minmax_normalize(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """按样本做 Min-Max 归一化到 [-1, 1]。"""
    lo, hi = signal.min(), signal.max()
    return (2 * (signal - lo) / (hi - lo + eps) - 1.0).astype(np.float32)


def apply_normalize(signal: np.ndarray, method: str) -> np.ndarray:
    """按方法名统一调用归一化函数。"""
    if method == "zscore":
        return z_normalize(signal)
    if method == "minmax":
        return minmax_normalize(signal)
    raise ValueError(f"unknown normalize method: {method}")


# ---------------------------------------------------------------------
# 切分
# ---------------------------------------------------------------------

def segment_by_rpeak(
    signal: np.ndarray,
    rpeaks: Iterable[int],
    win_before: int,
    win_after: int,
) -> Tuple[np.ndarray, List[int]]:
    """以每个 R 峰为中心截取固定窗口。

    Args:
        signal: 形状 `(n,)` 的一维信号。
        rpeaks: R 峰位置（样本点下标）。
        win_before, win_after: R 峰前/后各取多少采样点。

    Returns:
        segments: 形状 `(num_beats, win_before + win_after)` 的 `float32`。
        kept_indices: 实际被保留的 `rpeaks` 索引（边缘越界会被丢弃）。
    """
    if signal.ndim != 1:
        raise ValueError(f"segment_by_rpeak expects 1D signal, got {signal.shape}")
    n = signal.shape[0]
    win = win_before + win_after
    out: List[np.ndarray] = []
    kept: List[int] = []
    for i, r in enumerate(rpeaks):
        start = r - win_before
        end = r + win_after
        if start < 0 or end > n:
            continue
        out.append(signal[start:end])
        kept.append(i)
    if not out:
        return np.zeros((0, win), dtype=np.float32), kept
    return np.stack(out).astype(np.float32), kept


def segment_by_window(
    signal: np.ndarray,
    fs: float,
    window_sec: float,
    stride_sec: float | None = None,
) -> np.ndarray:
    """按固定时长对信号做滑动切分（扩展接口）。

    Args:
        signal: 一维信号。
        fs: 采样率。
        window_sec: 每个窗口的时长 (秒)。
        stride_sec: 窗口步长 (秒)，默认等于 window_sec 即不重叠。

    Returns:
        形状 `(num_windows, win_len)` 的数组。
    """
    if signal.ndim != 1:
        raise ValueError(f"segment_by_window expects 1D signal, got {signal.shape}")
    if stride_sec is None:
        stride_sec = window_sec
    win = int(round(window_sec * fs))
    stride = int(round(stride_sec * fs))
    if win <= 0 or stride <= 0:
        raise ValueError("window_sec and stride_sec must be positive")
    n = signal.shape[0]
    if n < win:
        return np.zeros((0, win), dtype=np.float32)
    starts = np.arange(0, n - win + 1, stride)
    out = np.stack([signal[s : s + win] for s in starts])
    return out.astype(np.float32)
