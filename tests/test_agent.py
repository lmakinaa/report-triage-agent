"""The agent loop. Concepts: tool calling, least privilege, step limits, logging."""
import json

from triage.agent import make_tools, run_agent
from triage.models import Case
from triage.retrieval import PolicyIndex
from tests.helpers import DATA, FakeLLM

CASE = Case(id="A1", market="SG", report_reason="spam", transcript="Send 50 dollars to my PayPal",
            viewer_reports=9, creator_id="cr_777")
CREATORS = {"cr_777": {"prior_strikes": 2, "account_age_days": 40, "agency": None}}


def tools():
    return make_tools(PolicyIndex.from_markdown(DATA / "policy.md"), CREATORS)


def tool(name, **args):
    return json.dumps({"tool": name, "args": args})


def final(label="SPAM_SCAM", conf=0.92, section="P1"):
    return json.dumps({"final": {"label": label, "confidence": conf, "reason": "r",
                                 "evidence": "e", "policy_section": section}})


def test_tools_work_on_their_own():
    t = tools()
    assert set(t) == {"search_policy", "get_creator_history"}
    assert "[P1]" in t["search_policy"]("paypal money")
    assert json.loads(t["get_creator_history"]("cr_777"))["prior_strikes"] == 2
    assert t["get_creator_history"]("nobody") == "Unknown creator."


def test_tool_then_final():
    llm = FakeLLM([tool("search_policy", query="paypal"), tool("get_creator_history", creator_id="cr_777"), final()])
    res = run_agent(CASE, llm, tools())
    assert res.decision.label == "SPAM_SCAM" and res.decision.case_id == "A1"
    assert "TOOL search_policy RESULT" in llm.calls[1][1], "feed the tool result back to the model"
    assert '"prior_strikes": 2' in llm.calls[2][1]
    assert [s.get("tool") for s in res.log] == ["search_policy", "get_creator_history", None]
    assert res.log[-1] == {"step": 3, "final": True}


def test_unknown_tool_is_refused_not_run():
    llm = FakeLLM([tool("ban_creator", creator_id="cr_777"), final()])
    res = run_agent(CASE, llm, tools())
    assert "ERROR: unknown tool ban_creator" in llm.calls[1][1]
    assert res.log[0]["ok"] is False and res.decision.label == "SPAM_SCAM"


def test_crashing_tool_does_not_crash_the_loop():
    def boom(**_):
        raise RuntimeError("database down")
    llm = FakeLLM([tool("search_policy", query="x"), final()])
    res = run_agent(CASE, llm, {"search_policy": boom})
    assert "ERROR: database down" in llm.calls[1][1]
    assert res.decision.label == "SPAM_SCAM"


def test_garbage_reply_gets_a_nudge():
    llm = FakeLLM(["let me think...", final()])
    res = run_agent(CASE, llm, tools())
    assert "ERROR" in llm.calls[1][1] and res.log[0]["ok"] is False


def test_step_limit():
    llm = FakeLLM([tool("search_policy", query="x")] * 3)
    res = run_agent(CASE, llm, tools(), max_steps=3)
    assert res.decision.label == "NEEDS_REVIEW"
    assert len(llm.calls) == 3 and len(res.log) == 3
