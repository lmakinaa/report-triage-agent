"""Evaluate. Concepts: precision, recall, F1, overkill, leakage, confusion matrix."""
import pytest

from triage.evaluate import evaluate
from triage.models import Case, Decision, Metrics


def case(i, label):
    return Case(id=f"E{i}", market="SG", report_reason="x", transcript="t", viewer_reports=0,
                creator_id="c", label=label)


def dec(i, label, conf=0.9):
    return Decision(case_id=f"E{i}", label=label, confidence=conf)


CASES = [case(1, "SPAM_SCAM"), case(2, "HARASSMENT"), case(3, "SAFE"), case(4, "SAFE"),
         case(5, "DANGEROUS_ACTIVITY"), case(6, "SAFE"), case(7, "GIFT_MANIPULATION")]
DECISIONS = [
    dec(1, "SPAM_SCAM"),            # tp
    dec(2, "SPAM_SCAM"),            # tp (wrong category, still caught)
    dec(3, "HARASSMENT"),           # fp  -> overkill
    dec(4, "SAFE"),                 # tn
    dec(5, "SAFE"),                 # fn  -> leakage
    dec(6, "NEEDS_REVIEW", 0.0),    # reviewed
    dec(7, "GIFT_MANIPULATION"),    # tp
    Decision(case_id="UNKNOWN", label="SAFE", confidence=0.9),  # no gold case: ignored
]


def test_counts():
    m = evaluate(DECISIONS, CASES)
    assert isinstance(m, Metrics)
    assert (m.n_cases, m.n_reviewed, m.tp, m.fp, m.fn, m.tn) == (7, 1, 3, 1, 1, 1)
    assert (m.overkill, m.leakage) == (1, 1)


def test_rates():
    m = evaluate(DECISIONS, CASES)
    assert m.precision == pytest.approx(0.75)
    assert m.recall == pytest.approx(0.75)
    assert m.f1 == pytest.approx(0.75)
    assert m.review_rate == pytest.approx(1 / 7)


def test_confusion_matrix():
    m = evaluate(DECISIONS, CASES)
    assert m.confusion["HARASSMENT"]["SPAM_SCAM"] == 1
    assert m.confusion["SAFE"]["NEEDS_REVIEW"] == 1
    assert m.confusion["DANGEROUS_ACTIVITY"]["SAFE"] == 1
    assert sum(sum(row.values()) for row in m.confusion.values()) == 7


def test_zero_denominators_do_not_crash():
    m = evaluate([dec(4, "SAFE")], CASES)
    assert (m.precision, m.recall, m.f1) == (0.0, 0.0, 0.0)
    empty = evaluate([], CASES)
    assert empty.n_cases == 0 and empty.review_rate == 0.0
