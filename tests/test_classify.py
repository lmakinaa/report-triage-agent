"""Structured decisions. Concepts: prompt structure, delimiters, JSON output, validate + retry."""
import json

import pytest

from triage.classify import build_prompt, classify_case, load_prompt, parse_decision
from triage.models import Case
from tests.helpers import FakeLLM, ok_reply

CASE = Case(id="T1", market="SG", report_reason="spam", transcript="Send money to my PayPal now",
            viewer_reports=4, creator_id="cr_9", label="SPAM_SCAM")


def test_prompt_structure():
    system, user = build_prompt(CASE)
    assert system == load_prompt("v1")
    for field in ("T1", "SG", "spam", "4"):
        assert field in user
    assert "<case>" in user and "</case>" in user
    inside = user.split("<case>")[1].split("</case>")[0]
    assert "Send money to my PayPal now" in inside, "the transcript goes inside the <case> tags"
    assert "<policy>" not in user, "no policy block without policy_context"


def test_prompt_with_policy_context():
    _, user = build_prompt(CASE, policy_context="[P1] Scams\nNo payment links.")
    inside = user.split("<policy>")[1].split("</policy>")[0]
    assert "No payment links." in inside


@pytest.mark.parametrize("text", [
    ok_reply("SPAM_SCAM", 0.93, "P1"),
    "```json\n" + ok_reply("SPAM_SCAM", 0.93, "P1") + "\n```",
    "```\n" + ok_reply("SPAM_SCAM", 0.93, "P1") + "\n```",
])
def test_parse_valid_replies(text):
    d = parse_decision(text, "T1")
    assert (d.case_id, d.label, d.confidence, d.policy_section) == ("T1", "SPAM_SCAM", 0.93, "P1")


@pytest.mark.parametrize("text", [
    "I think this is spam.",
    json.dumps({"label": "SPAMMY", "confidence": 0.9}),
    json.dumps({"label": "SAFE", "confidence": 1.7}),
    json.dumps({"label": "NEEDS_REVIEW", "confidence": 0.5}),
    json.dumps({"confidence": 0.5}),
    json.dumps(["SAFE", 0.9]),
])
def test_parse_rejects_bad_replies(text):
    with pytest.raises(ValueError):
        parse_decision(text, "T1")


def test_happy_path_calls_llm_once():
    llm = FakeLLM([ok_reply("SPAM_SCAM", 0.95, "P1")])
    d = classify_case(CASE, llm)
    assert d.label == "SPAM_SCAM" and d.case_id == "T1"
    assert len(llm.calls) == 1


def test_retries_once_with_the_error():
    llm = FakeLLM(["not json", ok_reply("SPAM_SCAM", 0.8)])
    d = classify_case(CASE, llm)
    assert d.label == "SPAM_SCAM"
    assert len(llm.calls) == 2
    first_user, retry_user = llm.calls[0][1], llm.calls[1][1]
    assert retry_user.startswith(first_user), "keep the original prompt on retry"
    assert "invalid" in retry_user.lower(), "tell the model what was wrong"


def test_gives_up_after_two_bad_replies():
    llm = FakeLLM(["nope", "still nope"])
    d = classify_case(CASE, llm)
    assert d.label == "NEEDS_REVIEW" and d.confidence == 0.0
    assert len(llm.calls) == 2, "never more than two calls per case"
