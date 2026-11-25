"""Shared data contracts passed between pipeline stages.

Every stage passes these objects around, so the tests rely on their exact fields.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field

LABELS = ("SAFE", "SPAM_SCAM", "HARASSMENT", "GIFT_MANIPULATION", "DANGEROUS_ACTIVITY")
Label = Literal["SAFE", "SPAM_SCAM", "HARASSMENT", "GIFT_MANIPULATION", "DANGEROUS_ACTIVITY"]
NEEDS_REVIEW = "NEEDS_REVIEW"


class Case(BaseModel):
    """One reported LIVE session snippet."""
    id: str
    market: str
    report_reason: str
    transcript: str = Field(min_length=1)
    viewer_reports: int = Field(ge=0)
    creator_id: str
    label: Optional[Label] = None  # the gold label; present in the golden set only


class Decision(BaseModel):
    """What the system decided about one case."""
    case_id: str
    label: Literal["SAFE", "SPAM_SCAM", "HARASSMENT", "GIFT_MANIPULATION", "DANGEROUS_ACTIVITY", "NEEDS_REVIEW"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    evidence: str = ""
    policy_section: Optional[str] = None

    @property
    def violating(self) -> bool:
        return self.label not in ("SAFE", NEEDS_REVIEW)


class Metrics(BaseModel):
    """Evaluation results for a set of decisions against gold labels."""
    n_cases: int          # decisions matched to a gold label
    n_reviewed: int       # of those, decisions labelled NEEDS_REVIEW
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    overkill: int         # = fp: acceptable content wrongly flagged
    leakage: int          # = fn: violating content that got through
    review_rate: float    # n_reviewed / n_cases
    confusion: dict[str, dict[str, int]]  # confusion[gold_label][predicted_label] = count
