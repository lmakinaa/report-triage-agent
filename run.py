"""Run the whole pipeline live against a real LLM and write a report.

Set three environment variables for any OpenAI-compatible API, for example:
    export LLM_BASE_URL=https://api.openai.com/v1
    export LLM_API_KEY=sk-...
    export LLM_MODEL=gpt-4o-mini
Then:
    python run.py --prompt v1            # plain classifier
    python run.py --prompt v2 --rag      # improved prompt + retrieved policy
    python run.py --agent                # the tool-using agent
Writes out/decisions.jsonl and out/report.md, and prints the metrics.
"""
import argparse
import json
import os
from pathlib import Path

from triage.agent import make_tools, run_agent
from triage.classify import classify_case
from triage.evaluate import evaluate
from triage.http_llm import HttpLLM
from triage.loader import load_cases
from triage.report import build_report, cost_estimate, value_estimate
from triage.retrieval import PolicyIndex, policy_context
from triage.robustness import with_retries
from triage.routing import route, tune_auto_threshold

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="v1")
    ap.add_argument("--rag", action="store_true")
    ap.add_argument("--agent", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only run the first N cases")
    ap.add_argument("--price-in", type=float, default=0.15, help="USD per 1M input tokens")
    ap.add_argument("--price-out", type=float, default=0.60, help="USD per 1M output tokens")
    a = ap.parse_args()

    llm = HttpLLM(os.environ["LLM_BASE_URL"], os.environ["LLM_API_KEY"], os.environ["LLM_MODEL"])
    cases, errors = load_cases(ROOT / "data" / "cases.jsonl")
    for e in errors:
        print("skipped:", e)
    cases = cases[: a.limit] if a.limit else cases
    index = PolicyIndex.from_markdown(ROOT / "data" / "policy.md")
    creators = json.loads((ROOT / "data" / "creators.json").read_text())
    tools = make_tools(index, creators)

    decisions = []
    for c in cases:
        if a.agent:
            d = with_retries(lambda: run_agent(c, llm, tools).decision)
        else:
            ctx = policy_context(index.search(c.transcript, k=2)) if a.rag else ""
            d = with_retries(lambda: classify_case(c, llm, policy_context=ctx, prompt_version=a.prompt))
        decisions.append(d)
        mark = "ok " if d.label == c.label else "XX "
        print(f"{mark}{c.id} gold={c.label:<18} pred={d.label:<18} conf={d.confidence:.2f} -> {route(d)}")

    m = evaluate(decisions, cases)
    value = value_estimate(cases_per_week=600, manual_minutes=4, review_rate=m.review_rate)
    cost = cost_estimate(llm.usage, a.price_in, a.price_out)
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    (out / "decisions.jsonl").write_text("\n".join(d.model_dump_json() for d in decisions) + "\n")
    report = build_report(m, value, cost, narrator=llm)
    (out / "report.md").write_text(report + "\n")
    print("\n" + report)
    print(f"\nSuggested auto-action threshold for 95% precision: {tune_auto_threshold(decisions, cases)}")


if __name__ == "__main__":
    main()
