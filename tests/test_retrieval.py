"""Retrieval (RAG). Concepts: chunking by section, TF-IDF / BM25 ranking, grounding with citations."""
import json

from triage.classify import classify_case
from triage.models import Case
from triage.retrieval import PolicyIndex, Section, policy_context, tokenize
from tests.helpers import DATA, FakeLLM, ok_reply

INDEX = None


def index():
    global INDEX
    INDEX = INDEX or PolicyIndex.from_markdown(DATA / "policy.md")
    return INDEX


def test_sections_are_parsed():
    secs = index().sections
    assert [s.id for s in secs] == ["P1", "P2", "P3", "P4", "P5"]
    assert secs[0].title == "Scams and spam"
    assert "PayPal" in secs[0].text and "##" not in secs[0].text


def test_tokenize():
    assert tokenize("The PayPal link, NOW!") == ["paypal", "link", "now"]


def test_search_finds_the_right_section():
    assert index().search("send money to my paypal link outside the app", k=1)[0].id == "P1"
    assert index().search("calls the viewer a stupid ugly loser", k=1)[0].id == "P2"
    assert index().search("gift roses and I will follow you back", k=1)[0].id == "P3"
    assert index().search("driving on the highway while streaming", k=1)[0].id == "P4"


def test_search_limits_and_empty_queries():
    assert len(index().search("gift", k=1)) == 1
    assert len(index().search("money gift harassment driving", k=3)) == 3
    assert index().search("") == []
    assert index().search("the and of") == []
    assert index().search("zzzqqq") == []


def test_policy_context_format():
    s = [Section(id="P9", title="Test", text="Body text")]
    assert policy_context(s) == "[P9] Test\nBody text"


def test_retrieved_policy_reaches_the_prompt():
    case = Case(id="R1", market="SG", report_reason="spam", transcript="pay my PayPal",
                viewer_reports=3, creator_id="c")
    ctx = policy_context(index().search(case.transcript, k=2))
    llm = FakeLLM([ok_reply("SPAM_SCAM", 0.9, "P1")])
    d = classify_case(case, llm, policy_context=ctx)
    assert "[P1] Scams and spam" in llm.calls[0][1]
    assert d.policy_section == "P1"
