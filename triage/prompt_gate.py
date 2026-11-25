"""Treat prompts like code: version them and block regressions."""
from pydantic import BaseModel
from triage.models import Metrics

class GateResult(BaseModel):
    passed: bool
    reasons: list[str]

def regression_gate(baseline: Metrics, candidate: Metrics, max_precision_drop: float=0.0, max_recall_drop: float=0.02, max_review_rate: float=0.2) -> GateResult:
    """Decide whether a candidate prompt may replace the baseline.

Fail (passed=False) for each rule broken, adding one reason string per rule:
  - candidate.precision < baseline.precision - max_precision_drop  -> reason mentions "precision"
  - candidate.recall    < baseline.recall    - max_recall_drop     -> reason mentions "recall"
  - candidate.review_rate > max_review_rate                        -> reason mentions "review"
Compare with a tiny tolerance (1e-9) so equal numbers never fail.
Pass with reasons=[] when no rule is broken."""
    eps = 1e-09
    reasons = []
    if candidate.precision < baseline.precision - max_precision_drop - eps:
        reasons.append(f'precision dropped from {baseline.precision:.3f} to {candidate.precision:.3f} (allowed drop: {max_precision_drop:.3f})')
    if candidate.recall < baseline.recall - max_recall_drop - eps:
        reasons.append(f'recall dropped from {baseline.recall:.3f} to {candidate.recall:.3f} (allowed drop: {max_recall_drop:.3f})')
    if candidate.review_rate > max_review_rate + eps:
        reasons.append(f'review rate {candidate.review_rate:.3f} is above the maximum of {max_review_rate:.3f}')
    return GateResult(passed=not reasons, reasons=reasons)
