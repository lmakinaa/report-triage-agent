"""Prompt iteration. Concepts: prompt versioning, regression testing, shipping gates."""
from triage.classify import build_prompt, load_prompt
from triage.models import Case, Metrics
from triage.prompt_gate import regression_gate

CASE = Case(id="T1", market="SG", report_reason="spam", transcript="hi", viewer_reports=0, creator_id="c")


def m(p, r, review=0.0):
    return Metrics(n_cases=10, n_reviewed=0, tp=0, fp=0, fn=0, tn=0, precision=p, recall=r, f1=0,
                   overkill=0, leakage=0, review_rate=review, confusion={})


def test_you_wrote_a_v2_prompt():
    v1, v2 = load_prompt("v1"), load_prompt("v2")
    assert v2.strip() and v2 != v1, "create prompts/v2.txt with your improved prompt"
    assert build_prompt(CASE, prompt_version="v2")[0] == v2


def test_v2_covers_the_basics():
    v2 = load_prompt("v2").lower()
    for label in ("safe", "spam_scam", "harassment", "gift_manipulation", "dangerous_activity"):
        assert label in v2, f"v2 must still list the label {label.upper()}"
    assert "json" in v2, "v2 must still demand JSON output"


def test_equal_metrics_pass():
    assert regression_gate(m(0.9, 0.8), m(0.9, 0.8)).passed


def test_precision_drop_fails():
    g = regression_gate(m(0.95, 0.80), m(0.93, 0.90))
    assert not g.passed and any("precision" in r for r in g.reasons)


def test_small_recall_drop_allowed_big_one_fails():
    assert regression_gate(m(0.9, 0.80), m(0.9, 0.79)).passed
    g = regression_gate(m(0.9, 0.80), m(0.9, 0.70))
    assert not g.passed and any("recall" in r for r in g.reasons)


def test_too_many_reviews_fails_and_reasons_accumulate():
    g = regression_gate(m(0.95, 0.9), m(0.90, 0.7, review=0.35))
    assert not g.passed and len(g.reasons) == 3
