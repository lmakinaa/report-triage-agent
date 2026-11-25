"""Load and validate the reported cases."""
import json
from pathlib import Path
from pydantic import ValidationError
from triage.models import Case

def load_cases(path: str | Path) -> tuple[list[Case], list[str]]:
    """
    Read a JSONL file of cases (one JSON object per line).
    Return (valid_cases, errors), in file order.
    - Blank lines are skipped silently.
    - A line that is not valid JSON, or fails Case validation, becomes an error.
    - A repeated id is an error; the first occurrence is kept.
    - Every error message starts with "line <n>: " (1-based line number).
    Never raise on a bad line: one broken row must not stop the whole batch.
    """
    cases, errors, seen = [], [], set()
    with open(path, encoding="utf-8") as f:
        for n, raw in enumerate(f, start=1):
            if not raw.strip():
                continue
            try:
                case = Case.model_validate_json(raw)
            except ValidationError as e:
                errors.append(f"line {n}: {_short(e)}")
                continue
            if case.id in seen:
                errors.append(f"line {n}: duplicate id {case.id!r}")
                continue
            seen.add(case.id)
            cases.append(case)
    return cases, errors

def _short(e: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(map(str, err['loc'])) or 'json'}: {err['msg']}"
        for err in e.errors()
    )