"""Test doubles. A FakeLLM replays scripted replies so tests never need an API key."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self.responses:
            raise AssertionError("FakeLLM ran out of scripted replies: your code called the LLM too many times")
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r if isinstance(r, str) else json.dumps(r)


def ok_reply(label="SAFE", confidence=0.9, section=None):
    return json.dumps({"label": label, "confidence": confidence, "reason": "test",
                       "evidence": "quote", "policy_section": section})
