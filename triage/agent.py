"""A tool-using agent loop with a step limit and a log."""
import json
from pathlib import Path
from pydantic import BaseModel
from triage.classify import build_prompt, parse_decision
from triage.models import NEEDS_REVIEW, Case, Decision
from triage.retrieval import PolicyIndex, policy_context
AGENT_PROMPT = Path(__file__).resolve().parent.parent / 'prompts' / 'agent.txt'

class AgentResult(BaseModel):
    decision: Decision
    log: list[dict]

def make_tools(index: PolicyIndex, creators: dict) -> dict:
    """Return the agent's read-only tools as {name: function}.

- "search_policy"(query: str) -> str: policy_context(index.search(query, k=2)),
  or "No matching policy." when nothing matches.
- "get_creator_history"(creator_id: str) -> str: json.dumps(creators[creator_id]),
  or "Unknown creator." when the id is missing."""
    def search_policy(query: str) -> str:
        found = index.search(query, k=2)
        return policy_context(found) if found else 'No matching policy.'

    def get_creator_history(creator_id: str) -> str:
        return json.dumps(creators[creator_id]) if creator_id in creators else 'Unknown creator.'
    return {'search_policy': search_policy, 'get_creator_history': get_creator_history}

def run_agent(case: Case, llm, tools: dict, max_steps: int=5) -> AgentResult:
    """Let the LLM investigate with tools, then decide.

system = the text of prompts/agent.txt; transcript starts as build_prompt(case)[1]
(plus the creator id: "Creator id: <id>"). Each step: reply = llm.complete(system, transcript).

Parse reply as JSON:
  {"tool": name, "args": {...}}:
      name in tools -> obs = tools[name](**args); a tool exception -> obs = "ERROR: <exception>"
      name not in tools -> obs = "ERROR: unknown tool <name>. Allowed: <comma-separated names>"
      (the unknown tool is never executed)
      append "\\n\\nTOOL <name> RESULT:\\n<obs>" to transcript
      log {"step": i, "tool": name, "args": args, "ok": <True unless an ERROR>}
  {"final": {...}}:
      decision = parse_decision(json.dumps(final), case.id);
      if that raises ValueError, treat it like invalid JSON below.
      log {"step": i, "final": True}; return
  anything else / invalid JSON:
      append "\\n\\nERROR: reply with a single JSON object." ; log {"step": i, "ok": False}
After max_steps replies without a final decision, return
Decision(label="NEEDS_REVIEW", confidence=0.0, reason="max steps reached").
Steps are numbered from 1."""
    system = AGENT_PROMPT.read_text(encoding='utf-8')
    transcript = build_prompt(case)[1] + f'\nCreator id: {case.creator_id}'
    log = []
    for i in range(1, max_steps + 1):
        reply = llm.complete(system, transcript)
        try:
            msg = json.loads(reply)
        except json.JSONDecodeError:
            msg = None
        if isinstance(msg, dict) and 'final' in msg:
            try:
                decision = parse_decision(json.dumps(msg['final']), case.id)
            except ValueError:
                decision = None
            if decision:
                log.append({'step': i, 'final': True})
                return AgentResult(decision=decision, log=log)
        elif isinstance(msg, dict) and 'tool' in msg:
            name, args = msg['tool'], msg.get('args', {})
            ok = True
            if name not in tools:
                obs = f"ERROR: unknown tool {name}. Allowed: {', '.join(tools)}"
                ok = False
            else:
                try:
                    obs = tools[name](**args)
                except Exception as e:
                    obs = f'ERROR: {e}'
                    ok = False
            transcript += f'\n\nTOOL {name} RESULT:\n{obs}'
            log.append({'step': i, 'tool': name, 'args': args, 'ok': ok})
            continue
        transcript += '\n\nERROR: reply with a single JSON object.'
        log.append({'step': i, 'ok': False})
    decision = Decision(case_id=case.id, label=NEEDS_REVIEW, confidence=0.0, reason='max steps reached')
    return AgentResult(decision=decision, log=log)
