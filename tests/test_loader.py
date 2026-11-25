"""Load the cases. Concepts: structured data, validation, never let one bad row kill a batch."""
from triage.loader import load_cases
from triage.models import Case
from tests.helpers import DATA, FIXTURES


def test_loads_the_golden_set():
    cases, errors = load_cases(DATA / "cases.jsonl")
    assert errors == [], f"the golden set should load cleanly, got: {errors}"
    assert len(cases) == 40
    assert all(isinstance(c, Case) for c in cases)
    assert cases[0].id == "C001" and cases[-1].id == "C040", "keep file order"


def test_bad_rows_become_errors_not_crashes():
    cases, errors = load_cases(FIXTURES / "bad_cases.jsonl")
    assert [c.id for c in cases] == ["X1", "X6"]
    assert len(errors) == 5, f"expected 5 errors (lines 2, 3, 4, 5, 8), got {errors}"


def test_error_messages_name_the_line():
    _, errors = load_cases(FIXTURES / "bad_cases.jsonl")
    lines = sorted(int(e.split(":")[0].replace("line", "").strip()) for e in errors)
    assert lines == [2, 3, 4, 5, 8], f"each error must start with 'line <n>:', got {errors}"


def test_duplicate_id_is_reported():
    _, errors = load_cases(FIXTURES / "bad_cases.jsonl")
    assert any(e.startswith("line 5:") and "duplicate" in e.lower() for e in errors)
