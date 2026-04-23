"""基于规则的 Agent 决策模块。

根据 `RiskResult` 中的风险等级输出结构化响应策略，不依赖学习算法，
保证行为可解释、可审计。输出字段：

    - timestamp   发生决策的 UTC 时间；
    - risk_score  连续风险值；
    - risk_level  "low" / "medium" / "high"；
    - action      机器可读的动作码；
    - message     人类可读的提示文本；
    - top_class   置信度最高的类别标签；
    - class_probs 类别概率字典；
    - meta        任意附加字段（患者 ID、设备号等）。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from ..risk.assessor import RiskResult, _as_dict


class RuleBasedAgent:
    """把风险等级映射到响应策略。"""

    def __init__(self, actions: Dict[str, str], messages: Dict[str, str]):
        for k in ("low", "medium", "high"):
            if k not in actions or k not in messages:
                raise ValueError(f"agent config missing key '{k}'")
        self.actions = dict(actions)
        self.messages = dict(messages)

    @classmethod
    def from_config(cls, cfg) -> "RuleBasedAgent":
        return cls(
            actions=_as_dict(cfg.agent.actions),
            messages=_as_dict(cfg.agent.messages),
        )

    def decide(
        self,
        result: RiskResult,
        meta: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        level = result.level
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "risk_score": round(result.score, 6),
            "risk_level": level,
            "action": self.actions[level],
            "message": self.messages[level],
            "top_class": result.top_class,
            "top_prob": round(result.top_prob, 6),
            "class_probs": {k: round(v, 6) for k, v in result.class_probs.items()},
            "meta": dict(meta or {}),
        }
