"""Routing. Concepts: confidence thresholds, human-in-the-loop, tuning for precision."""
from triage.models import Case, Decision
from triage.routing import route, tune_auto_threshold


def d(label, conf, i="Q"):
    return Decision(case_id=i, label=label, confidence=conf)


def test_route_table():
    assert route(d("SPAM_SCAM", 0.95)) == "auto_action"
    assert route(d("SPAM_SCAM", 0.90)) == "auto_action", "threshold is inclusive"
    assert route(d("SPAM_SCAM", 0.75)) == "human_review"
    assert route(d("SAFE", 0.80)) == "no_action"
    assert route(d("SAFE", 0.40)) == "human_review"
    assert route(d("HARASSMENT", 0.30)) == "human_review"
    assert route(d("NEEDS_REVIEW", 0.0)) == "human_review"


def test_custom_thresholds():
    assert route(d("SPAM_SCAM", 0.85), auto_threshold=0.8) == "auto_action"
    assert route(d("SAFE", 0.55), review_threshold=0.5) == "no_action"


def test_tune_threshold():
    cases = [Case(id=f"Q{i}", market="SG", report_reason="x", transcript="t", viewer_reports=0,
                  creator_id="c", label=lab) for i, lab in enumerate(
                      ["SPAM_SCAM", "SPAM_SCAM", "SAFE", "HARASSMENT", "SAFE"])]
    decisions = [
        d("SPAM_SCAM", 0.97, "Q0"),   # correct
        d("SPAM_SCAM", 0.92, "Q1"),   # correct
        d("HARASSMENT", 0.82, "Q2"),  # overkill at 0.82
        d("HARASSMENT", 0.88, "Q3"),  # correct
        d("SAFE", 0.99, "Q4"),        # not violating, never auto-actioned
    ]
    assert tune_auto_threshold(decisions, cases, min_precision=0.95) == 0.85
    assert tune_auto_threshold(decisions, cases, min_precision=0.7) == 0.5
    assert tune_auto_threshold([d("SPAM_SCAM", 0.99, "Q2")], cases) is None


def test_needs_review_is_never_auto_actioned_even_when_confident():
    assert route(d("NEEDS_REVIEW", 0.99)) == "human_review"


def test_tune_threshold_ignores_needs_review_decisions():
    cases = [Case(id="Q0", market="SG", report_reason="x", transcript="t", viewer_reports=0,
                  creator_id="c", label="SAFE")]
    assert tune_auto_threshold([d("NEEDS_REVIEW", 0.99, "Q0")], cases) is None
