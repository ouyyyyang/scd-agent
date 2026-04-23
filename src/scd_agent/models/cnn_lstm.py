"""1D-CNN + LSTM 心律失常分类模型。

- CNN 块：多层 `Conv1d + BatchNorm1d + ReLU + MaxPool1d` 提取局部波形特征；
- LSTM 块：对降采样后的特征序列建模时间依赖，支持双向；
- 分类头：全连接 → Dropout → 输出 logits。
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    """Conv1d → BN → ReLU → MaxPool1d。"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


class CNN_LSTM(nn.Module):
    """1D-CNN + LSTM 分类器。

    Args:
        in_channels: 输入通道数（单导联 ECG 为 1，多模态可 >1）。
        num_classes: 输出类别数。
        conv_channels: 各卷积层的输出通道数列表。
        kernel_size: 卷积核大小。
        lstm_hidden: LSTM 隐层维度。
        lstm_layers: LSTM 堆叠层数。
        bidirectional: 是否双向 LSTM。
        dropout: 分类头的 Dropout 概率。
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 5,
        conv_channels: Sequence[int] = (32, 64, 128),
        kernel_size: int = 7,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        blocks: List[nn.Module] = []
        prev = in_channels
        for ch in conv_channels:
            blocks.append(_ConvBlock(prev, ch, kernel_size))
            prev = ch
        self.cnn = nn.Sequential(*blocks)

        self.lstm = nn.LSTM(
            input_size=prev,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        directions = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * directions, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 形状 `(batch, channels, seq_len)`。

        Returns:
            形状 `(batch, num_classes)` 的 logits。
        """
        if x.ndim != 3:
            raise ValueError(f"expect (B, C, L) input, got shape {tuple(x.shape)}")
        feats = self.cnn(x)                    # (B, C', L')
        feats = feats.transpose(1, 2)          # (B, L', C')
        out, _ = self.lstm(feats)              # (B, L', H*dirs)
        pooled = out.mean(dim=1)               # 时间维全局平均
        return self.head(pooled)


def build_model(cfg) -> CNN_LSTM:
    """根据 `Config.model` 构造模型。"""
    m = cfg.model
    return CNN_LSTM(
        in_channels=int(m.in_channels),
        num_classes=int(m.num_classes),
        conv_channels=tuple(m.conv_channels),
        kernel_size=int(m.kernel_size),
        lstm_hidden=int(m.lstm_hidden),
        lstm_layers=int(m.lstm_layers),
        bidirectional=bool(m.bidirectional),
        dropout=float(m.dropout),
    )
