"""Prove the value. Concepts: time saved, token cost, numbers from code, narrative from the LLM."""
import pytest

from triage.models import Metrics
from triage.report import build_report, cost_estimate, value_estimate
from tests.helpers import FakeLLM

M = Metrics(n_cases=40, n_reviewed=4, tp=20, fp=2, fn=4, tn=10, precision=20 / 22, recall=20 / 24,
            f1=0.87, overkill=2, leakage=4, review_rate=0.1, confusion={})


def test_value_estimate():
    v = value_estimate(cases_per_week=600, manual_minutes=4, review_rate=0.1, spot_check_minutes=0.5)
    assert v["before_hours"] == pytest.approx(40.0)
    assert v["after_hours"] == pytest.approx(600 * (0.1 * 4 + 0.9 * 0.5) / 60)
    assert v["hours_saved"] == pytest.approx(v["before_hours"] - v["after_hours"])


def test_cost_estimate():
    usage = [{"prompt_tokens": 400_000, "completion_tokens": 50_000},
             {"prompt_tokens": 600_000, "completion_tokens": 50_000}]
    assert cost_estimate(usage, usd_per_m_input=0.15, usd_per_m_output=0.60) == pytest.approx(0.21)
    assert cost_estimate([], 1, 1) == 0.0


def test_report_table_is_exact():
    r = build_report(M, {"hours_saved": 31.5}, 0.4249)
    assert r.startswith("# Weekly triage report\n\n| Metric | Value |\n| --- | --- |\n")
    for row in ("| Precision | 90.9% |", "| Recall | 83.3% |", "| Overkill | 2 |", "| Leakage | 4 |",
                "| Review rate | 10.0% |", "| Hours saved per week | 31.5 |", "| LLM cost | $0.42 |"):
        assert row in r, f"missing row: {row}"
    assert "## Summary" not in r


def test_narrator_writes_words_code_keeps_numbers():
    narrator = FakeLLM(["  Precision was 99%! Great week.  "])
    r = build_report(M, {"hours_saved": 31.5}, 0.42, narrator=narrator)
    assert "| Precision | 90.9% |" in r, "the code table stays, even if the narrator gets numbers wrong"
    assert r.endswith("## Summary\n\nPrecision was 99%! Great week.")
    assert "90.9%" in narrator.calls[0][1], "give the narrator the real numbers"
