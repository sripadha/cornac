"""Offline tests for the Week 4 provider hardening — no network, no Ollama, no API key.

Ollama: the provider talks HTTP through an httpx.Client, and httpx ships a
MockTransport that routes every request to a plain Python function instead of a
socket. We swap the provider's client for one built on that, so a handler in the test
plays the daemon: it can inspect the exact JSON we sent and answer with any body or
status code we like. That pins down what goes on the wire, how the reply is parsed,
and what happens when the daemon fails.

Anthropic: the SDK does not open a connection when the client is constructed, so we
build a provider with a dummy key and feed `_from_anthropic` a hand-made response
object shaped like the SDK's (types.SimpleNamespace stands in for its pydantic models).
The outbound translation, `_to_anthropic`, needs no client at all, so those tests build
the provider with `__new__` and skip `__init__` entirely.
"""

import json
import types

import httpx
import pytest

from cornac.core.messages import Message, ToolCall, ToolResult, Usage
from cornac.providers.anthropic import AnthropicProvider
from cornac.providers.ollama import OllamaProvider

# --- helpers -----------------------------------------------------------------


def ollama_reply(text: str = "hi", **extra) -> dict:
    """A minimal /api/chat response body, with any extra top-level keys merged in."""
    body = {"model": "fake", "message": {"role": "assistant", "content": text}, "done": True}
    body.update(extra)
    return body


def make_provider(handler, **kwargs) -> OllamaProvider:
    """An OllamaProvider whose HTTP goes to `handler(request)` instead of a daemon."""
    provider = OllamaProvider(**kwargs)
    provider._client = httpx.Client(transport=httpx.MockTransport(handler))
    provider._sleep = lambda seconds: None  # never actually wait in a unit test
    return provider


def one_turn(provider: OllamaProvider) -> Message:
    return provider.complete([Message.user("hi")], tools=[])


# --- Ollama: what goes on the wire -------------------------------------------


def test_request_carries_num_ctx_seed_and_temperature():
    sent: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent["path"] = request.url.path
        sent["body"] = json.loads(request.content)
        return httpx.Response(200, json=ollama_reply())

    provider = make_provider(handler, model="m", num_ctx=4096, seed=7, temperature=0.3)
    one_turn(provider)

    assert sent["path"] == "/api/chat"
    body = sent["body"]
    assert body["model"] == "m"
    assert body["stream"] is False
    assert body["options"] == {"temperature": 0.3, "num_ctx": 4096, "seed": 7}


def test_defaults_are_a_generous_context_and_a_pinned_seed():
    sent: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return httpx.Response(200, json=ollama_reply())

    one_turn(make_provider(handler))

    # 2048 (Ollama's default) silently truncates an agent conversation; 8192 does not.
    # temperature 0 alone is not determinism, hence the seed.
    assert sent["options"] == {"temperature": 0.0, "num_ctx": 8192, "seed": 42}


def test_a_full_conversation_is_sent_in_ollama_wire_shape():
    # One of each neutral message kind, plus a tool schema — and the exact JSON that
    # must reach the daemon for it. This pins every translation rule in _to_ollama
    # (system stays a role, tool results are their own role and carry tool_name) and
    # the {"type": "function", "function": {...}} wrapping of tools.
    sent: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return httpx.Response(200, json=ollama_reply("It is noon."))

    call = ToolCall(id="call_0_get_current_time", name="get_current_time",
                    arguments={"timezone": "UTC"})
    messages = [
        Message.system("Be brief."),
        Message.user("What time is it?"),
        Message(role="assistant", text="", tool_calls=[call]),
        Message.from_tool_result(
            ToolResult(tool_call_id=call.id, name="get_current_time", content="12:00 UTC")
        ),
    ]
    schema = {
        "type": "object",
        "properties": {"timezone": {"type": "string"}},
        "required": ["timezone"],
    }
    tools = [{"name": "get_current_time", "description": "Current time in a timezone.",
              "input_schema": schema}]

    msg = make_provider(handler).complete(messages, tools)

    assert msg.text == "It is noon."
    assert sent["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "What time is it?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "get_current_time", "arguments": {"timezone": "UTC"}}}
            ],
        },
        {"role": "tool", "content": "12:00 UTC", "tool_name": "get_current_time"},
    ]
    assert sent["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Current time in a timezone.",
                "parameters": schema,
            },
        }
    ]


# --- Ollama: what comes back -------------------------------------------------


def test_usage_and_stop_reason_are_parsed_from_the_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=ollama_reply("done", prompt_eval_count=120, eval_count=30, done_reason="stop"),
        )

    msg = one_turn(make_provider(handler))

    assert msg.role == "assistant"
    assert msg.text == "done"
    assert msg.usage == Usage(input_tokens=120, output_tokens=30)
    assert msg.usage.total_tokens == 150
    assert msg.stop_reason == "stop"


def test_missing_usage_keys_count_as_zero_and_missing_stop_reason_is_none():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=ollama_reply("bare"))

    msg = one_turn(make_provider(handler))

    assert msg.usage == Usage(input_tokens=0, output_tokens=0)
    assert msg.stop_reason is None


def test_null_usage_keys_also_count_as_zero():
    # A key that is present but null must not smuggle a None into Usage: the agent
    # adds usage up across the run, and int + None would crash it mid-loop.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=ollama_reply("bare", prompt_eval_count=None, eval_count=None)
        )

    msg = one_turn(make_provider(handler))

    assert msg.usage == Usage(input_tokens=0, output_tokens=0)
    assert Usage() + msg.usage == Usage()  # exactly the sum the agent loop performs


def test_tool_calls_still_parse_alongside_usage():
    body = ollama_reply("", prompt_eval_count=10, eval_count=5, done_reason="stop")
    body["message"]["tool_calls"] = [
        {"function": {"name": "get_current_time", "arguments": {"timezone": "UTC"}}}
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    msg = one_turn(make_provider(handler))

    assert msg.wants_tools
    assert msg.tool_calls == [
        ToolCall(id="call_0_get_current_time", name="get_current_time", arguments={"timezone": "UTC"})
    ]
    assert msg.usage == Usage(input_tokens=10, output_tokens=5)


# --- Ollama: retries ---------------------------------------------------------


def scripted_handler(statuses: list[int], calls: list[int]):
    """A handler that answers with each status in turn, recording every call."""

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        status = statuses.pop(0)
        if status == 200:
            return httpx.Response(200, json=ollama_reply("ok"))
        return httpx.Response(status, text="daemon says no")

    return handler


def test_a_500_is_retried_once_and_then_succeeds():
    calls: list[int] = []
    sleeps: list[float] = []
    provider = make_provider(scripted_handler([500, 200], calls), backoff=0.5)
    provider._sleep = sleeps.append

    msg = one_turn(provider)

    assert msg.text == "ok"
    assert len(calls) == 2
    assert sleeps == [0.5]  # exactly one wait, of `backoff` seconds


def test_three_consecutive_500s_give_up_with_the_last_error():
    calls: list[int] = []
    sleeps: list[float] = []
    # max_retries counts the attempts AFTER the first (the Anthropic SDK's meaning):
    # 2 retries = 3 attempts in total.
    provider = make_provider(scripted_handler([500, 503, 502], calls), backoff=0.5, max_retries=2)
    provider._sleep = sleeps.append

    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        one_turn(provider)

    assert excinfo.value.response.status_code == 502  # the *last* failure, not the first
    assert len(calls) == 3
    assert sleeps == [0.5, 1.0]  # exponential: backoff, 2*backoff — and no sleep after the last try


def test_max_retries_zero_means_exactly_one_attempt():
    # 0 retries is not "0 attempts": the request is still made once, and a failure
    # is raised at once with no waiting. The 200 queued behind the 500 is never seen.
    calls: list[int] = []
    sleeps: list[float] = []
    provider = make_provider(scripted_handler([500, 200], calls), max_retries=0)
    provider._sleep = sleeps.append

    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        one_turn(provider)

    assert excinfo.value.response.status_code == 500
    assert len(calls) == 1
    assert sleeps == []


def test_a_4xx_is_raised_immediately_without_retrying():
    calls: list[int] = []
    sleeps: list[float] = []
    provider = make_provider(scripted_handler([400, 200], calls))
    provider._sleep = sleeps.append

    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        one_turn(provider)

    assert excinfo.value.response.status_code == 400
    assert len(calls) == 1  # the 200 queued behind it was never requested
    assert sleeps == []


def test_a_dropped_connection_is_retried_like_a_500():
    calls: list[int] = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("connection refused")  # daemon still loading the model
        return httpx.Response(200, json=ollama_reply("ok"))

    provider = make_provider(handler, backoff=0.25)
    provider._sleep = sleeps.append

    msg = one_turn(provider)

    assert msg.text == "ok"
    assert len(calls) == 2
    assert sleeps == [0.25]


def test_persistent_connection_failure_reraises_the_transport_error():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ConnectError("connection refused")

    provider = make_provider(handler, max_retries=2)

    with pytest.raises(httpx.ConnectError):
        one_turn(provider)

    assert len(calls) == 3  # the first attempt plus two retries


# --- Ollama: the daemon's reason survives into the error ---------------------


def test_error_message_carries_ollamas_own_reason():
    # The lesson from a real afternoon: httpx said "500 Internal Server Error";
    # Ollama's body said "cudaMalloc failed: out of memory". The second one is the
    # diagnosis; the first is a shrug.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "cudaMalloc failed: out of memory"})

    provider = make_provider(handler, max_retries=0)
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        one_turn(provider)

    assert "500" in str(excinfo.value)
    assert "cudaMalloc failed: out of memory" in str(excinfo.value)
    assert excinfo.value.response.status_code == 500  # .response is still there for callers


def test_a_4xx_reason_is_kept_too():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'qwen9000' not found"})

    provider = make_provider(handler)
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        one_turn(provider)

    assert "404" in str(excinfo.value)
    assert "model 'qwen9000' not found" in str(excinfo.value)


def test_a_non_json_error_body_falls_back_to_its_text():
    # Not every failure comes from Ollama itself — a proxy in front of it may answer
    # with an HTML page. Then the body text is the best explanation we have.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="<html>Bad Gateway</html>")

    provider = make_provider(handler, max_retries=0)
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        one_turn(provider)

    assert "502" in str(excinfo.value)
    assert "Bad Gateway" in str(excinfo.value)


# --- Anthropic ---------------------------------------------------------------


def fake_anthropic_response(**overrides) -> types.SimpleNamespace:
    """A response object shaped like the SDK's, using SimpleNamespace for attribute access."""
    fields = {
        "content": [
            types.SimpleNamespace(type="text", text="Let me check."),
            types.SimpleNamespace(
                type="tool_use", id="toolu_1", name="get_current_time", input={"timezone": "UTC"}
            ),
        ],
        "usage": types.SimpleNamespace(input_tokens=50, output_tokens=12),
        "stop_reason": "tool_use",
    }
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


def test_anthropic_client_is_built_with_sdk_retries():
    provider = AnthropicProvider(api_key="dummy-key-no-network")
    assert provider._client.max_retries == 3


def test_from_anthropic_parses_text_tool_calls_usage_and_stop_reason():
    provider = AnthropicProvider(api_key="dummy-key-no-network")

    msg = provider._from_anthropic(fake_anthropic_response())

    assert msg.role == "assistant"
    assert msg.text == "Let me check."
    assert msg.tool_calls == [
        ToolCall(id="toolu_1", name="get_current_time", arguments={"timezone": "UTC"})
    ]
    assert msg.usage == Usage(input_tokens=50, output_tokens=12)
    assert msg.stop_reason == "tool_use"


def test_from_anthropic_final_answer_has_end_turn_and_no_tool_calls():
    provider = AnthropicProvider(api_key="dummy-key-no-network")
    response = fake_anthropic_response(
        content=[types.SimpleNamespace(type="text", text="It is noon.")],
        usage=types.SimpleNamespace(input_tokens=80, output_tokens=6),
        stop_reason="end_turn",
    )

    msg = provider._from_anthropic(response)

    assert msg.text == "It is noon."
    assert not msg.wants_tools
    assert msg.usage.total_tokens == 86
    assert msg.stop_reason == "end_turn"


def test_from_anthropic_tolerates_a_response_with_no_usage_or_stop_reason():
    provider = AnthropicProvider(api_key="dummy-key-no-network")
    response = types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text="hi")])

    msg = provider._from_anthropic(response)

    assert msg.text == "hi"
    assert msg.usage is None  # "unknown", not a made-up zero
    assert msg.stop_reason is None


# --- Anthropic: what goes on the wire ----------------------------------------


def bare_anthropic_provider() -> AnthropicProvider:
    """A provider with no client: _to_anthropic is pure translation and needs none."""
    return AnthropicProvider.__new__(AnthropicProvider)


def test_to_anthropic_omits_an_empty_assistant_turn():
    # The Messages API rejects {"role": "assistant", "content": []} and whitespace-only
    # text blocks. Such a turn is reachable (a reply of "" or one made only of block
    # types we ignore) and would break the NEXT request, when the history is re-sent.
    # It carries nothing, so it is dropped; its neighbours must be untouched.
    messages = [
        Message.system("Be brief."),
        Message.user("first"),
        Message(role="assistant", text=""),        # nothing at all
        Message.user("second"),
        Message(role="assistant", text="   \n"),   # whitespace only: same thing
        Message.user("third"),
        Message(role="assistant", text="a real answer"),
    ]

    system, api_messages = bare_anthropic_provider()._to_anthropic(messages)

    assert system == "Be brief."
    assert api_messages == [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
        {"role": "user", "content": "third"},
        {"role": "assistant", "content": [{"type": "text", "text": "a real answer"}]},
    ]


def test_to_anthropic_keeps_an_assistant_turn_that_only_calls_tools():
    # Empty text next to a tool call is the normal shape of a tool turn, not an
    # empty turn: the tool_use block is the content.
    call = ToolCall(id="toolu_1", name="get_current_time", arguments={"timezone": "UTC"})
    messages = [
        Message.user("time?"),
        Message(role="assistant", text="", tool_calls=[call]),
        Message.from_tool_result(
            ToolResult(tool_call_id="toolu_1", name="get_current_time", content="noon", is_error=True)
        ),
    ]

    _, api_messages = bare_anthropic_provider()._to_anthropic(messages)

    assert api_messages == [
        {"role": "user", "content": "time?"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "get_current_time",
                 "input": {"timezone": "UTC"}}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "noon",
                 "is_error": True}
            ],
        },
    ]
