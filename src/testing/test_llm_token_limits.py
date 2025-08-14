import types

import pytest

from llm_utils import call_chat_completion


class _FakeChatMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeChatMessage(content)


class _FakeChatResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]
        self.usage = {'prompt_tokens': 1, 'completion_tokens': 1}


class _FakeResponsesOutputText:
    def __init__(self, value: str):
        self.value = value


class _FakeResponsesContent:
    def __init__(self, text_value: str):
        self.text = _FakeResponsesOutputText(text_value)


class _FakeResponsesItem:
    def __init__(self, text_value: str):
        self.content = [_FakeResponsesContent(text_value)]


class _FakeResponsesResponse:
    def __init__(self, text: str):
        # Simulate the Responses API output format used by llm_utils
        self.output = [_FakeResponsesItem(text)]
        self.output_text = text
        self.usage = {'input_tokens': 1, 'output_tokens': 1}


class _ChatAPI:
    def __init__(self, behavior):
        self._behavior = behavior
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._behavior(self, kwargs)


class _ResponsesAPI:
    def __init__(self, behavior):
        self._behavior = behavior
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._behavior(self, kwargs)


class _FakeOpenAIClient:
    def __init__(self, chat_behavior=None, responses_behavior=None):
        self.chat = types.SimpleNamespace(completions=_ChatAPI(chat_behavior))
        self.responses = _ResponsesAPI(responses_behavior)


def test_chat_adaptive_retry_on_token_error():
    # First call raises token-limit error; second succeeds
    def behavior(self_api, kwargs):
        # On first call, raise error simulating token limit exceeded
        if len(self_api.calls) == 1:
            raise Exception("max_tokens is too large for this model")
        # On second call, ensure tokens reduced by ~25%
        assert 'max_completion_tokens' in kwargs
        return _FakeChatResponse("ok")

    client = _FakeOpenAIClient(chat_behavior=behavior)

    content, meta = call_chat_completion(
        client,
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-4o-mini",
        max_tokens=16000,  # request above ceiling; will clamp then adaptively retry
    )

    # Two calls due to adaptive retry
    calls = client.chat.completions.calls
    assert len(calls) == 2
    # First call uses clamped ceiling (<= 12000)
    first_tokens = calls[0]['max_completion_tokens']
    assert first_tokens <= 12000
    # Second call uses reduced tokens (<= 75% of the first)
    second_tokens = calls[1]['max_completion_tokens']
    assert second_tokens <= int(first_tokens * 0.75)
    assert content == "ok"


def test_responses_adaptive_retry_on_token_error():
    # First call raises token-limit error; second succeeds
    def behavior(self_api, kwargs):
        if len(self_api.calls) == 1:
            raise Exception("max_tokens is too large: max_output_tokens exceeds limit")
        # Ensure we normalized to max_output_tokens and reduced
        assert 'max_output_tokens' in kwargs
        return _FakeResponsesResponse("ok-resp")

    client = _FakeOpenAIClient(responses_behavior=behavior)

    content, meta = call_chat_completion(
        client,
        messages=[{"role": "user", "content": "ping"}],
        model="gpt-5-mini",
        max_tokens=12000,  # request above conservative ceiling
    )

    calls = client.responses.calls
    assert len(calls) == 2
    first_tokens = calls[0]['max_output_tokens']
    # First call is clamped to model ceiling (<= 6000 per registry)
    assert first_tokens <= 6000
    second_tokens = calls[1]['max_output_tokens']
    # Second call should be <= 75% of first (and <= ceiling)
    assert second_tokens <= int(first_tokens * 0.75)
    assert second_tokens <= 6000
    assert content == "ok-resp"


def test_chat_initial_clamp_without_retry():
    # No error path: ensure initial clamp is applied and only one call
    def behavior(self_api, kwargs):
        assert 'max_completion_tokens' in kwargs
        # Should be clamped to <= 12000 for gpt-4o-mini
        assert kwargs['max_completion_tokens'] <= 12000
        return _FakeChatResponse("hello")

    client = _FakeOpenAIClient(chat_behavior=behavior)
    content, meta = call_chat_completion(
        client,
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-4o-mini",
        max_tokens=50000,
    )
    assert len(client.chat.completions.calls) == 1
    assert content == "hello"


