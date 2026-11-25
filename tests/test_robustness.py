"""Robustness. Concepts: backoff, retryable vs fatal errors, consistency, prompt injection."""
import pytest

from triage.classify import build_prompt, classify_case
from triage.errors import LLMError, RateLimitError, TransientError
from triage.loader import load_cases
from triage.models import Case
from triage.robustness import flip_rate, with_retries
from tests.helpers import DATA, FakeLLM, ok_reply


class Flaky:
    def __init__(self, errors, result="ok"):
        self.errors, self.result, self.calls = list(errors), result, 0

    def __call__(self):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.result


def test_retries_with_exponential_backoff():
    sleeps = []
    fn = Flaky([RateLimitError("429"), TransientError("503"), RateLimitError("429")])
    assert with_retries(fn, sleep=sleeps.append) == "ok"
    assert fn.calls == 4 and sleeps == [1.0, 2.0, 4.0]


def test_gives_up_after_max_attempts():
    sleeps = []
    fn = Flaky([TransientError("x")] * 10)
    with pytest.raises(TransientError):
        with_retries(fn, max_attempts=3, base_delay=0.5, sleep=sleeps.append)
    assert fn.calls == 3 and sleeps == [0.5, 1.0]


def test_fatal_errors_are_not_retried():
    sleeps = []
    fn = Flaky([LLMError("401 bad key", status=401)])
    with pytest.raises(LLMError):
        with_retries(fn, sleep=sleeps.append)
    assert fn.calls == 1 and sleeps == []


def test_flip_rate():
    case = Case(id="F1", market="SG", report_reason="x", transcript="t", viewer_reports=0, creator_id="c")
    llm = FakeLLM([ok_reply("SAFE"), ok_reply("SAFE"), ok_reply("HARASSMENT")])
    assert flip_rate(case, llm, runs=3) == pytest.approx(1 / 3)
    steady = FakeLLM([ok_reply("SAFE")] * 4)
    assert flip_rate(case, steady, runs=4) == 0.0


def test_transcript_cannot_break_out_of_the_case_tags():
    cases, _ = load_cases(DATA / "cases.jsonl")
    attack = next(c for c in cases if "</case>" in c.transcript)
    _, user = build_prompt(attack)
    assert user.count("<case>") == 1 and user.count("</case>") == 1
    assert "Ignore previous rules" in user.split("<case>")[1].split("</case>")[0], \
        "keep the attack text visible inside the tags so the model can judge it"
