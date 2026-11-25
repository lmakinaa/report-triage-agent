"""Confidence routing and human-in-the-loop."""
from triage.models import NEEDS_REVIEW, Case, Decision
AUTO_ACTION, HUMAN_REVIEW, NO_ACTION = ('auto_action', 'human_review', 'no_action')

def route(decision: Decision, auto_threshold: float=0.9, review_threshold: float=0.6) -> str:
    '''Send each decision to the right queue.

- label NEEDS_REVIEW                              -> "human_review"
- violating and confidence >= auto_threshold      -> "auto_action"
- violating and confidence >= review_threshold    -> "human_review"
- SAFE and confidence >= review_threshold         -> "no_action"
- anything with confidence < review_threshold     -> "human_review"'''
    if decision.label == NEEDS_REVIEW or decision.confidence < review_threshold:
        return HUMAN_REVIEW
    if not decision.violating:
        return NO_ACTION
    return AUTO_ACTION if decision.confidence >= auto_threshold else HUMAN_REVIEW

def tune_auto_threshold(decisions: list[Decision], cases: list[Case], min_precision: float=0.95, candidates: tuple[float, ...]=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99)) -> float | None:
    """Pick the LOWEST threshold whose auto-actioned set is precise enough.

For each candidate t (lowest first): take decisions that are violating with confidence >= t
and whose case has a gold label. Precision = share of those whose gold label is not SAFE.
Return the first t with at least one such decision and precision >= min_precision.
Return None if no candidate qualifies."""
    gold = {case.id: case.label for case in cases if case.label is not None}
    for t in sorted(candidates):
        picked = [d for d in decisions if d.violating and d.confidence >= t and d.case_id in gold]
        if not picked:
            continue
        correct = sum(1 for d in picked if gold[d.case_id] != 'SAFE')
        precision = correct / len(picked)
        if precision >= min_precision:
            return t
    return None
