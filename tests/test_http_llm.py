"""Talk to the API. Concepts: REST, headers, JSON bodies, status codes, token usage."""
import pytest

from triage.errors import LLMError, RateLimitError, TransientError
from triage.http_llm import HttpLLM


class FakePost:
    def __init__(self, status, body):
        self.status, self.body, self.calls = status, body, []

    def __call__(self, url, headers, body, timeout):
        self.calls.append({"url": url, "headers": headers, "body": body, "timeout": timeout})
        return self.status, self.body


def reply(text, usage=True):
    body = {"choices": [{"message": {"role": "assistant", "content": text}}]}
    if usage:
        body["usage"] = {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}
    return body


def make(post):
    return HttpLLM("https://api.example.com/v1/", "sk-test", "test-model", post=post)


def test_sends_a_correct_request():
    post = FakePost(200, reply("hi"))
    make(post).complete("SYS", "USER")
    call = post.calls[0]
    assert call["url"] == "https://api.example.com/v1/chat/completions", "no double slash"
    assert call["headers"]["Authorization"] == "Bearer sk-test"
    assert call["headers"]["Content-Type"] == "application/json"
    assert call["body"]["model"] == "test-model"
    assert call["body"]["temperature"] == 0
    assert call["body"]["messages"] == [{"role": "system", "content": "SYS"},
                                        {"role": "user", "content": "USER"}]


def test_returns_text_and_records_usage():
    llm = make(FakePost(200, reply("hello there")))
    assert llm.complete("s", "u") == "hello there"
    assert llm.usage == [{"prompt_tokens": 120, "completion_tokens": 30}]


def test_missing_usage_is_fine():
    llm = make(FakePost(200, reply("ok", usage=False)))
    assert llm.complete("s", "u") == "ok"
    assert llm.usage == []


@pytest.mark.parametrize("status,exc", [(429, RateLimitError), (500, TransientError),
                                        (503, TransientError), (401, LLMError), (400, LLMError)])
def test_status_codes_map_to_errors(status, exc):
    with pytest.raises(exc) as info:
        make(FakePost(status, {"error": "nope"})).complete("s", "u")
    assert info.value.status == status


def test_rate_limit_is_not_a_plain_llm_error():
    with pytest.raises(RateLimitError):
        make(FakePost(429, {})).complete("s", "u")
    with pytest.raises(LLMError) as info:
        make(FakePost(401, {})).complete("s", "u")
    assert not isinstance(info.value, (RateLimitError, TransientError)), "401 must not be retryable"


def test_unexpected_200_shape_raises():
    with pytest.raises(LLMError):
        make(FakePost(200, {"weird": True})).complete("s", "u")
