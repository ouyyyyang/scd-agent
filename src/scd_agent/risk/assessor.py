"""风险评估模块。

把分类器的概率分布映射成连续的风险值，并按阈值划分为 low/medium/high。

评分公式：
    risk = Σ_i  w_i  *  p_i  /  max(w)

其中 `w_i` 来自 `config.risk.class_weights`，恶性心律类别（V/F 等）权重较大。
对最大权重做归一化确保风险值始终落在 [0, 1]。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class RiskResult:
    score: float
    level: str  # "low" | "medium" | "high"
    top_class: str
    top_prob: float
    class_probs: Dict[str, float]


def _as_dict(maybe_config) -> Dict:
    """接受 Config 或原生 dict，统一返回 dict。"""
    if hasattr(maybe_config, "to_dict"):
        return maybe_config.to_dict()
    return dict(maybe_config)


class RiskAssessor:
    """加权风险评分 + 阈值分级。"""

    def __init__(
        self,
        classes: Sequence[str],
        class_weights: Dict[str, float],
        low_threshold: float,
        medium_threshold: float,
    ):
        self.classes: List[str] = list(classes)
        self.weights = np.asarray(
            [float(class_weights.get(c, 0.0)) for c in self.classes], dtype=np.float32
        )
        if low_threshold >= medium_threshold:
            raise ValueError(
                f"low_threshold ({low_threshold}) must be < medium_threshold ({medium_threshold})"
            )
        self.low_threshold = float(low_threshold)
        self.medium_threshold = float(medium_threshold)

    @classmethod
    def from_config(cls, cfg, classes: Sequence[str] | None = None) -> "RiskAssessor":
        """从 Config 构造；若显式传入 `classes`（例如来自 checkpoint.meta），
        则使用该顺序而非 cfg 中的顺序，避免训练/推理列序不一致导致权重错配。
        """
        classes = list(classes) if classes is not None else list(cfg.labels.classes)
        weights = _as_dict(cfg.risk.class_weights)
        thresholds = cfg.risk.thresholds
        return cls(
            classes=classes,
            class_weights=weights,
            low_threshold=float(thresholds.low),
            medium_threshold=float(thresholds.medium),
        )

    # ----------------------------- API -----------------------------

    def score(self, probs: np.ndarray) -> float:
        """根据概率向量计算标量风险值 (clamp 到 [0, 1])。"""
        probs = np.asarray(probs, dtype=np.float32)
        if probs.shape != self.weights.shape:
            raise ValueError(
                f"probs shape {probs.shape} does not match classes {self.weights.shape}"
            )
        raw = float(np.dot(probs, self.weights))
        denom = float(self.weights.max()) or 1.0
        return max(0.0, min(1.0, raw / denom))

    def grade(self, risk: float) -> str:
        if risk < self.low_threshold:
            return "low"
        if risk < self.medium_threshold:
            return "medium"
        return "high"

    def assess(self, probs: np.ndarray) -> RiskResult:
        probs = np.asarray(probs, dtype=np.float32).ravel()
        if probs.shape[0] != len(self.classes):
            raise ValueError(
                f"probs has {probs.shape[0]} values but there are {len(self.classes)} classes"
            )
        s = self.score(probs)
        level = self.grade(s)
        top_idx = int(np.argmax(probs))
        return RiskResult(
            score=s,
            level=level,
            top_class=self.classes[top_idx],
            top_prob=float(probs[top_idx]),
            class_probs={c: float(p) for c, p in zip(self.classes, probs)},
        )
