import numpy as np
import pytest

from scd_agent.data.preprocess import (
    apply_normalize,
    bandpass_filter,
    minmax_normalize,
    segment_by_rpeak,
    segment_by_window,
    z_normalize,
)


def test_bandpass_filter_preserves_shape_and_removes_high_freq():
    fs = 360.0
    t = np.arange(0, 4 * fs) / fs
    low = np.sin(2 * np.pi * 2 * t)
    high = np.sin(2 * np.pi * 80 * t)  # 80 Hz，带通外，应被显著衰减
    signal = (low + high).astype(np.float32)
    filt = bandpass_filter(signal, fs=fs, low=0.5, high=40.0, order=4)
    assert filt.shape == signal.shape
    # 滤波后与低频成分的 RMSE 应明显小于与原始 (low+high) 的 RMSE
    rmse_vs_low = np.sqrt(np.mean((filt - low) ** 2))
    rmse_vs_orig = np.sqrt(np.mean((filt - signal) ** 2))
    assert rmse_vs_low < 0.2
    assert rmse_vs_orig > rmse_vs_low


def test_z_and_minmax_normalize():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    z = z_normalize(x)
    assert abs(z.mean()) < 1e-5
    assert abs(z.std() - 1.0) < 1e-3

    m = minmax_normalize(x)
    assert abs(m.min() + 1.0) < 1e-5
    assert abs(m.max() - 1.0) < 1e-5

    assert np.allclose(apply_normalize(x, "zscore"), z)
    assert np.allclose(apply_normalize(x, "minmax"), m)
    with pytest.raises(ValueError):
        apply_normalize(x, "other")


def test_segment_by_rpeak_drops_out_of_bound_and_keeps_indices():
    signal = np.arange(100, dtype=np.float32)
    rpeaks = [5, 50, 95]  # 5 在前边界外（win_before=10），95 在后边界外
    segs, kept = segment_by_rpeak(signal, rpeaks, win_before=10, win_after=10)
    assert segs.shape == (1, 20)
    assert kept == [1]
    assert segs[0][0] == 40 and segs[0][-1] == 59


def test_segment_by_rpeak_empty():
    signal = np.arange(50, dtype=np.float32)
    segs, kept = segment_by_rpeak(signal, [], 5, 5)
    assert segs.shape == (0, 10)
    assert kept == []


def test_segment_by_window():
    signal = np.arange(1000, dtype=np.float32)
    fs = 100.0
    out = segment_by_window(signal, fs=fs, window_sec=2.0)  # win=200, stride=200
    assert out.shape == (5, 200)
    overlap = segment_by_window(signal, fs=fs, window_sec=2.0, stride_sec=1.0)
    assert overlap.shape[0] == 9  # 0..800，步长 100

    too_short = segment_by_window(np.zeros(10, dtype=np.float32), fs=100.0, window_sec=1.0)
    assert too_short.shape == (0, 100)
