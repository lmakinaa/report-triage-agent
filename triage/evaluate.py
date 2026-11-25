"""Measure decisions against gold labels."""
from collections import defaultdict
from triage.models import NEEDS_REVIEW, Case, Decision, Metrics

def evaluate(decisions: list[Decision], cases: list[Case]) -> Metrics:
    """Compute Metrics for decisions against the cases' gold labels.

- Only count decisions whose case_id matches a case that has a gold label: that's n_cases.
- "Violating" means any label other than SAFE.
- NEEDS_REVIEW decisions count in n_reviewed and in the confusion matrix, but are
  excluded from tp / fp / fn / tn.
- tp: predicted violating, gold violating (the category need not match)
  fp: predicted violating, gold SAFE        (overkill)
  fn: predicted SAFE, gold violating        (leakage)
  tn: predicted SAFE, gold SAFE
- precision = tp/(tp+fp), recall = tp/(tp+fn), f1 = 2PR/(P+R); use 0.0 when a denominator is 0.
- review_rate = n_reviewed / n_cases (0.0 if n_cases is 0).
- confusion[gold][predicted] = count, for every pair seen (gold and predicted are label strings)."""
    n_reviewed = tp = fp = tn = fn = 0

    gold = {c.id: c.label for c in cases if c.label is not None}
    analysed_decisions = [d for d in decisions if d.case_id in gold]
    confusion = defaultdict(lambda: defaultdict(int))
    n_cases = len(analysed_decisions)

    for d in analysed_decisions:
        if d.label == NEEDS_REVIEW:
            n_reviewed += 1
            confusion[gold[d.case_id]][d.label] += 1
            continue

        if gold[d.case_id] != "SAFE":
            if d.label != "SAFE":
                tp += 1
            else:
                fn += 1
        else:
            if d.label != "SAFE":
                fp += 1
            else:
                tn += 1
        confusion[gold[d.case_id]][d.label] += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    review_rate = n_reviewed / n_cases if n_cases else 0.0
    return Metrics(n_cases=n_cases, n_reviewed=n_reviewed, tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall, f1=f1, overkill=fp, leakage=fn, review_rate=review_rate, confusion={g: dict(row) for g, row in confusion.items()})
