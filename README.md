# LIVE Case Triage Agent

An LLM pipeline that triages reported livestream moderation cases. It classifies each case, grounds the decision in a written policy, decides whether a human needs to look, and measures how well it did, all in plain Python with no framework.

It is built around a question moderation teams actually face: **how much reviewer time can we save without letting violations through?** So the pipeline does more than call a model. It measures precision and recall against a gold set, blocks prompt changes that make things worse, routes uncertain cases to humans, and turns the results into a weekly time and cost report.

> The dataset and policy in `data/` are fictional and written for this project. They are not any real platform's policy or data.

## What it does

```
          cases.jsonl                      policy.md
              │                                │
        ┌─────▼─────┐                   ┌──────▼──────┐
        │  loader   │                   │  retrieval  │  TF-IDF search over policy sections
        └─────┬─────┘                   └──────┬──────┘
              │   case                         │ relevant sections
        ┌─────▼────────────────────────────────▼─────┐
        │  classify  (or agent: LLM + tools + steps) │  prompt → LLM → validated Decision
        └─────┬──────────────────────────────────────┘
              │ Decision(label, confidence, reason, evidence, policy_section)
     ┌────────┼───────────────┬──────────────────┐
┌────▼───┐ ┌──▼───────┐ ┌─────▼─────┐   ┌────────▼───────┐
│evaluate│ │ routing  │ │ prompt    │   │ report         │
│P/R, FP,│ │ auto /   │ │ regression│   │ hours saved,   │
│FN, conf│ │ human /  │ │ gate      │   │ LLM cost,      │
│ matrix │ │ no-action│ │           │   │ weekly summary │
└────────┘ └──────────┘ └───────────┘   └────────────────┘
```

| Module | Responsibility |
| --- | --- |
| [`triage/loader.py`](triage/loader.py) | Load and validate JSONL cases. A bad row becomes an error message and never stops the batch. |
| [`triage/http_llm.py`](triage/http_llm.py) | Minimal client for any OpenAI-compatible chat API, using only the standard library. It maps 429 to `RateLimitError` and 5xx to `TransientError`, and tracks token usage. |
| [`triage/classify.py`](triage/classify.py) | Builds the prompt, calls the model, parses and validates JSON into a `Decision`, and retries once on invalid output before falling back to `NEEDS_REVIEW`. |
| [`triage/evaluate.py`](triage/evaluate.py) | Precision, recall, F1, overkill (false positives), leakage (false negatives), review rate and a confusion matrix. |
| [`triage/prompt_gate.py`](triage/prompt_gate.py) | Regression gate that decides whether a new prompt may replace the old one. |
| [`triage/retrieval.py`](triage/retrieval.py) | Splits the policy by section and ranks sections with TF-IDF cosine similarity (RAG, no external services). |
| [`triage/agent.py`](triage/agent.py) | Tool-using agent loop with read-only tools, a step limit and a step log. |
| [`triage/routing.py`](triage/routing.py) | Confidence routing to auto-action, human review or no action, plus threshold tuning for a target precision. |
| [`triage/robustness.py`](triage/robustness.py) | Retry with exponential backoff, and a flip-rate check for answer consistency. |
| [`triage/report.py`](triage/report.py) | Time saved, token cost and a markdown weekly report. |

## Design decisions worth knowing about

- **Typed decisions.** The model's reply must parse into a pydantic `Decision`: label from a fixed set, confidence in [0, 1]. The model is not allowed to return `NEEDS_REVIEW` itself. That label belongs to the system, for when the model's output can't be trusted.
- **Prompt-injection defence.** The transcript is untrusted. It is wrapped in `<case>` tags, and any `<case>` or `</case>` inside it, in any letter case or spacing, is escaped so it cannot close the block and pass text off as instructions. The prompts also tell the model to treat tagged text as data. One case in the dataset (`C034`) is exactly this attack, and a test covers it.
- **Prompts are versioned like code.** `prompts/v1.txt` is the baseline and `prompts/v2.txt` adds per-label criteria, edge cases, few-shot examples and a data-not-instructions rule. A prompt only ships if `regression_gate` passes: no drop in precision, no more than 2 points of recall lost, and a review rate of 20% or less. Each broken rule gets its own reason.
- **Least-privilege agent.** The agent can only call tools you register (`search_policy`, `get_creator_history`). Both are read-only. A request for an unknown tool is refused and never executed, a crashing tool is reported back to the model instead of crashing the loop, and a step limit ends runaway loops with a `NEEDS_REVIEW` decision.
- **Numbers come from code.** The report table is computed deterministically. If an LLM writes the optional summary, it is appended after the table and never replaces it, so a model that gets a figure wrong can't change the numbers.
- **Failure handling.** Rate limits and 5xx errors retry with 1s, 2s and 4s backoff. A 401 or other client error fails immediately, because retrying a bad key doesn't help.
- **Everything is testable offline.** The LLM is passed in as an object with `complete(system, user) -> str`. Tests use a scripted `FakeLLM`, so the suite needs no API key and no network.

## Quick start

```bash
git clone <this repo> && cd triage-agent
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest
```

Requires Python 3.10 or newer. The only runtime dependency is `pydantic`.

## Run it against a real model

Any OpenAI-compatible API works (OpenAI, Groq, OpenRouter, a local Ollama server, and so on):

```bash
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_API_KEY=sk-...
export LLM_MODEL=gpt-4o-mini

python run.py --prompt v1 --limit 10   # baseline classifier on the first 10 cases
python run.py --prompt v2 --rag        # improved prompt plus retrieved policy sections
python run.py --agent                  # tool-using agent
```

Each run prints a per-case line (gold label, predicted label, confidence and routing queue), then the metrics. It also writes `out/decisions.jsonl` and `out/report.md`, and suggests the lowest auto-action confidence threshold that keeps precision at 95% or better on the labelled set.

Options: `--prompt NAME`, `--rag`, `--agent`, `--limit N`, `--price-in` and `--price-out` (USD per 1M tokens, used for the cost line in the report).

### The workflow this supports

1. Run the baseline (`--prompt v1`) and note the metrics.
2. Change the prompt, run again and compare the metrics.
3. Pass both `Metrics` objects to `regression_gate(baseline, candidate)`. Ship the new prompt only if it passes.
4. Use the tuned threshold to decide which decisions are safe to auto-action and which go to people.

## Data

`data/cases.jsonl` holds 40 labelled cases across six markets (ID, MY, PH, TH, VN, SG): 16 SAFE and 24 violating, spread over spam/scam, harassment, gift manipulation and dangerous activity. It deliberately includes hard negatives, such as a host warning viewers about a scam, and trash talk aimed at a game character, as well as an injection attempt. `data/policy.md` is the five-section rulebook the retrieval step searches, and `data/creators.json` is the creator history the agent can look up.

## Tests

```bash
python -m pytest              # all tests
python -m pytest tests/test_evaluate.py -v
```

There are 64 tests, with one file per module. They cover the loader's error handling, the HTTP client's status mapping, JSON parsing and the single retry, metric math including empty and zero-denominator cases, the regression gate, retrieval ranking, the agent loop (unknown tools, crashing tools, step limits), routing and threshold tuning, backoff timing, and the injection case. CI runs the suite on Python 3.10 and 3.12.

## Project layout

```
triage/        pipeline code (see the table above)
prompts/       versioned system prompts (v1, v2, agent)
data/          cases, policy and creator history (fictional)
tests/         offline tests with a scripted fake LLM
run.py         end-to-end runner for a live model
```

## Limitations

- The 40-case dataset is small and fictional. It is enough to exercise the pipeline and show the workflow, not to make claims about real moderation accuracy. I haven't published live-model metrics here, so run it against your own model to get numbers.
- Retrieval is TF-IDF over five sections, with no stemming or embeddings. That suits a short policy file. A longer policy would call for embeddings or a hybrid search.
- Text is lowercased and tokenised on ASCII letters and digits only, which would need changing for non-Latin transcripts.
- Confidence is whatever the model reports. It is not calibrated, so check threshold choices against labelled data, which is what `tune_auto_threshold` is for.
