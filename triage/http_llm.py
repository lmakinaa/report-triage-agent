"""Talk to an LLM over HTTP (any OpenAI-compatible chat completions API)."""
import json
import urllib.error
import urllib.request
from triage.errors import LLMError, RateLimitError, TransientError

def _default_post(url: str, headers: dict, body: dict, timeout: float) -> tuple[int, dict]:
    """Send a JSON POST and return (status_code, parsed_json_body)."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return (resp.status, json.loads(resp.read() or b'{}'))
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read() or b'{}')
        except json.JSONDecodeError:
            payload = {}
        return (e.code, payload)
    except (urllib.error.URLError, TimeoutError) as e:
        return (503, {'error': str(e)})

class HttpLLM:
    """A tiny client for any OpenAI-compatible API (OpenAI, Groq, OpenRouter, Ollama, ...)."""

    def __init__(self, base_url: str, api_key: str, model: str, post=None, timeout: float=30.0):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.post = post or _default_post
        self.timeout = timeout
        self.usage: list[dict] = []

    def complete(self, system: str, user: str) -> str:
        """
    Send one chat request and return the reply text.

    Request: POST {base_url}/chat/completions
    headers: {"Authorization": "Bearer <api_key>", "Content-Type": "application/json"}
    body:    {"model": <model>, "temperature": 0,
                "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}]}
    Call it as self.post(url, headers, body, self.timeout) -> (status, json_dict).

    Response handling:
    200 -> return body["choices"][0]["message"]["content"]; if body has "usage",
            append {"prompt_tokens": .., "completion_tokens": ..} to self.usage.
            A 200 without that path -> LLMError.
    429 -> raise RateLimitError(status=429)
    5xx -> raise TransientError(status=<code>)
    any other status -> raise LLMError(status=<code>)
    """
        status, res = self.post(
            self.base_url + "/chat/completions",
            {
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json"
            },
            {
                "model": self.model,
                "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]
            },
            self.timeout
        )
        match status:
            case 200:
                try:
                    content = res["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    raise LLMError(status=status, message="Malformed 200 response")
                usage = res.get("usage")
                if usage:
                    self.usage.append({
                        "prompt_tokens": usage["prompt_tokens"],
                        "completion_tokens": usage["completion_tokens"],
                    })
                return content
            case 429:
                raise RateLimitError(status=status, message="Rate limit error")
            case s if 500 <= s < 600:
                raise TransientError(status=s, message="Server Error")
            case _:
                raise LLMError(status=status, message="Unknown error")

