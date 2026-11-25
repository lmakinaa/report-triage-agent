"""Turn one case into a validated Decision (prompting + structured output)."""
import json
import re
from pathlib import Path
from pydantic import ValidationError
from triage.models import NEEDS_REVIEW, Case, Decision
PROMPTS_DIR = Path(__file__).resolve().parent.parent / 'prompts'

def load_prompt(version: str='v1') -> str:
    """Return the text of prompts/<version>.txt. Raise FileNotFoundError if missing."""
    return (PROMPTS_DIR / f'{version}.txt').read_text(encoding='utf-8')

def _neutralize(text: str, tag: str) -> str:
    """Escape any <tag> / </tag> in untrusted text (any case, any inner spacing) so it can't open or close our tags."""
    return re.sub(rf'<(\s*/?\s*{tag}\b[^>]*>)', r'&lt;\1', text, flags=re.IGNORECASE)

def build_prompt(case: Case, policy_context: str='', prompt_version: str='v1') -> tuple[str, str]:
    """Return (system, user) for one case.

system = load_prompt(prompt_version)
user must contain:
  - the case id, market, report_reason and viewer_reports
  - the transcript wrapped in <case> ... </case> tags (instructions stay outside the tags)
  - if policy_context is non-empty: it wrapped in <policy> ... </policy> tags
The user prompt must contain exactly one "<case>" and one "</case>",
even when the transcript itself contains those strings (neutralize them)."""
    system = load_prompt(prompt_version)
    safe_transcript = _neutralize(case.transcript, 'case')
    user = (
        f"case_id: {case.id}\n"
        f"market: {case.market}\n"
        f"report_reason: {case.report_reason}\n"
        f"viewer_reports: {case.viewer_reports}\n\n"
        f"<case>\n{safe_transcript}\n</case>"
    )
    if policy_context:
        user += f"\n\n<policy>\n{_neutralize(policy_context, 'policy')}\n</policy>"
    return system, user

def parse_decision(text: str, case_id: str) -> Decision:
    """Parse the model's reply into a Decision for case_id.

- Accept bare JSON or JSON wrapped in a ```json ... ``` (or ``` ... ```) fence.
- The JSON must be an object with label and confidence; reason, evidence, policy_section optional.
- label must be one of the five LABELS (the model may NOT return NEEDS_REVIEW itself).
- confidence must be a number from 0 to 1.
Raise ValueError with a short, readable message on any problem."""

    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    raw_json = m.group(1) if m else text.strip()
    try:
        decision_data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e.msg}") from e
    if not isinstance(decision_data, dict): 
        raise ValueError("expected a JSON object")
    decision_data["case_id"] = case_id
    if decision_data.get("label") == NEEDS_REVIEW:
        raise ValueError("model may not return NEEDS_REVIEW")
    try:
        return Decision.model_validate(decision_data)
    except ValidationError as e:
        err = e.errors()[0]
        raise ValueError(f"bad {err['loc'][0]}: {err['msg']}") from e

def classify_case(case: Case, llm, policy_context: str='', prompt_version: str='v1') -> Decision:
    """Classify one case with llm.complete(system, user) -> str.

    1. Build the prompt and call the LLM once.
    2. If parse_decision raises ValueError, call the LLM ONE more time with the same system
       prompt and the user prompt plus a blank line and
       "Your previous reply was invalid: <error>. Reply with valid JSON only."
    3. If the second reply is also invalid, return
       Decision(case_id=case.id, label="NEEDS_REVIEW", confidence=0.0, reason="invalid model output").
    Never call the LLM more than twice. Let LLM exceptions (RateLimitError etc.) propagate.
    """
    system, user = build_prompt(case, policy_context, prompt_version)
    try:
        return parse_decision(llm.complete(system, user), case.id)
    except ValueError as first_error:
        retry_user = f"{user}\n\nYour previous reply was invalid: {first_error}. Reply with valid JSON only."
    try:
        return parse_decision(llm.complete(system, retry_user), case.id)
    except ValueError:
        return Decision(case_id=case.id, label=NEEDS_REVIEW, confidence=0.0, reason="invalid model output")
