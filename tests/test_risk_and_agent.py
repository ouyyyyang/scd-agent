import numpy as np
import pytest

from scd_agent.agent.decision import RuleBasedAgent
from scd_agent.risk.assessor import RiskAssessor


@pytest.fixture
def assessor() -> RiskAssessor:
    return RiskAssessor(
        classes=["N", "S", "V", "F", "Q"],
        class_weights={"N": 0.0, "S": 0.4, "V": 1.0, "F": 0.8, "Q": 0.3},
        low_threshold=0.25,
        medium_threshold=0.60,
    )


def test_risk_score_bounds_and_grade(assessor: RiskAssessor):
    all_n = np.array([1.0, 0, 0, 0, 0], dtype=np.float32)
    all_v = np.array([0, 0, 1.0, 0, 0], dtype=np.float32)
    mixed = np.array([0.3, 0.4, 0.1, 0.1, 0.1], dtype=np.float32)

    r_n = assessor.assess(all_n)
    r_v = assessor.assess(all_v)
    r_m = assessor.assess(mixed)

    assert r_n.level == "low" and r_n.score == 0.0
    assert r_v.level == "high" and r_v.score == 1.0
    assert r_m.level in {"medium", "low"}
    assert 0.0 <= r_m.score <= 1.0
    assert r_v.top_class == "V"
    assert r_n.top_class == "N"


def test_risk_assessor_rejects_invalid_thresholds():
    with pytest.raises(ValueError):
        RiskAssessor(
            classes=["N", "V"], class_weights={"N": 0.0, "V": 1.0},
            low_threshold=0.6, medium_threshold=0.5,
        )


def test_risk_assessor_rejects_shape_mismatch(assessor: RiskAssessor):
    with pytest.raises(ValueError):
        assessor.assess(np.array([0.5, 0.5]))


def test_agent_decisions_cover_all_levels(assessor: RiskAssessor):
    agent = RuleBasedAgent(
        actions={"low": "normal_monitoring", "medium": "early_warning", "high": "emergency_response"},
        messages={"low": "ok", "medium": "warn", "high": "emergency"},
    )
    for probs, expect_level, expect_action in [
        (np.array([1.0, 0, 0, 0, 0]), "low", "normal_monitoring"),
        (np.array([0.3, 0.4, 0.1, 0.1, 0.1]), "medium", "early_warning"),
        (np.array([0, 0, 1.0, 0, 0]), "high", "emergency_response"),
    ]:
        decision = agent.decide(assessor.assess(probs.astype(np.float32)), meta={"x": 1})
        assert decision["risk_level"] == expect_level
        assert decision["action"] == expect_action
        assert "timestamp" in decision and decision["meta"] == {"x": 1}


def test_agent_requires_all_levels():
    with pytest.raises(ValueError):
        RuleBasedAgent(actions={"low": "a"}, messages={"low": "m", "medium": "m", "high": "m"})
