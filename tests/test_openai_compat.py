"""Offline tests for the OpenAI-compatible provider — no network, no server.

Same technique as the Ollama tests: the provider's httpx.Client is rebuilt on an
httpx.MockTransport, so a plain function in the test plays the server. It sees the
exact JSON we sent and answers with any body or status it likes. That pins down the
wire format (the whole point of this provider), how replies are parsed — including
the garbled-JSON case a local model produces now and then — and the retry behaviour,
which must match its sibling OllamaProvider.
"""

import json

import httpx
import pytest

from cornac.core.messages import Message, ToolCall, ToolResult, Usage
from cornac.providers.openai_compat import DEFAULT_BASE_URL, OpenAICompatibleProvider

# --- helpers -----------------------------------------------------------------


def openai_reply(
    text="hi",
    tool_calls=None,
    finish_reason="stop",
    usage=None,
    **extra,
) -> dict:
    """A minimal /chat/completions response body.

    `text=None` reproduces the real shape of a tools-only reply (content is null).
    `usage=None` means "send the usual counters"; pass {} to leave them out.
    """
    message: dict = {"role": "assistant", "content": text}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    body = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "fake",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
        if usage is None
        else usage,
    }
    body.update(extra)
    return body


def make_provider(handler, **kwargs) -> OpenAICompatibleProvider:
    """A provider whose HTTP goes to `handler(request)` instead of a server.

    The client is rebuilt on the mock transport but keeps the provider's headers,
    so the api_key -> Authorization behaviour is still what gets tested.
    """
    provider = OpenAICompatibleProvider(**kwargs)
    provider._client = httpx.Client(
        transport=httpx.MockTransport(handler), headers=provider._client.headers
    )
    provider._sleep = lambda seconds: None  # never actually wait in a unit test
    return provider


def recording_handler(reply: dict, sent: dict):
    """A handler that stores the request (path, headers, body) in `sent` and answers `reply`."""

    def handler(request: httpx.Request) -> httpx.Response:
        sent["method"] = request.method
        sent["url"] = str(request.url)
        sent["path"] = request.url.path
        sent["headers"] = dict(request.headers)
        sent["body"] = json.loads(request.content)
        return httpx.Response(200, json=reply)

    return handler


def one_turn(provider: OpenAICompatibleProvider, tools=()) -> Message:
    return provider.complete([Message.user("hi")], tools=list(tools))


SCHEMA = {
    "type": "object",
    "properties": {"timezone": {"type": "string"}},
    "required": ["timezone"],
}
TOOLS = [{"name": "get_current_time", "description": "Current time in a timezone.", "input_schema": SCHEMA}]


# --- what goes on the wire ---------------------------------------------------


def test_posts_to_chat_completions_under_the_base_url():
    sent: dict = {}
    provider = make_provider(recording_handler(openai_reply(), sent), base_url="http://srv:8000/v1", model="m")

    one_turn(provider)

    assert sent["method"] == "POST"
    assert sent["url"] == "http://srv:8000/v1/chat/completions"
    assert sent["body"]["model"] == "m"
    assert sent["body"]["messages"] == [{"role": "user", "content": "hi"}]


def test_a_trailing_slash_on_the_base_url_does_not_double_up():
    sent: dict = {}
    provider = make_provider(recording_handler(openai_reply(), sent), base_url="http://srv:8000/v1/")

    one_turn(provider)

    assert sent["url"] == "http://srv:8000/v1/chat/completions"
    assert provider.base_url == "http://srv:8000/v1"


def test_defaults_point_at_vllm_and_pin_temperature_zero():
    sent: dict = {}
    provider = make_provider(recording_handler(openai_reply(), sent))

    one_turn(provider)

    assert DEFAULT_BASE_URL == "http://localhost:8000/v1"
    assert sent["url"] == "http://localhost:8000/v1/chat/completions"
    assert sent["body"]["temperature"] == 0.0
    assert sent["body"]["model"] == ""  # always sent: the spec calls it required
    # No tools -> neither key. Several servers reject tool_choice without tools.
    assert "tools" not in sent["body"]
    assert "tool_choice" not in sent["body"]


def test_tools_are_wrapped_as_functions_and_tool_choice_is_auto():
    sent: dict = {}
    provider = make_provider(recording_handler(openai_reply(), sent))

    one_turn(provider, TOOLS)

    assert sent["body"]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Current time in a timezone.",
                "parameters": SCHEMA,
            },
        }
    ]
    assert sent["body"]["tool_choice"] == "auto"


def test_a_full_conversation_is_sent_in_openai_wire_shape():
    # One of each neutral message kind. The rules pinned here are the ones that differ
    # from Ollama: tool-call arguments become a JSON string, every call carries an id
    # and a type, and a tool result is matched by tool_call_id (no tool_name).
    sent: dict = {}
    provider = make_provider(recording_handler(openai_reply("It is noon."), sent))

    call = ToolCall(id="call_abc", name="get_current_time", arguments={"timezone": "UTC"})
    messages = [
        Message.system("Be brief."),
        Message.user("What time is it?"),
        Message(role="assistant", text="", tool_calls=[call]),
        Message.from_tool_result(
            ToolResult(tool_call_id="call_abc", name="get_current_time", content="12:00 UTC")
        ),
    ]

    msg = provider.complete(messages, TOOLS)

    assert msg.text == "It is noon."
    assert sent["body"]["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "What time is it?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "get_current_time", "arguments": '{"timezone": "UTC"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_abc", "content": "12:00 UTC"},
    ]


def test_an_assistant_turn_with_text_and_no_tool_calls_has_no_tool_calls_key():
    sent: dict = {}
    provider = make_provider(recording_handler(openai_reply(), sent))

    provider.complete([Message.user("q"), Message(role="assistant", text="a"), Message.user("q2")], [])

    assert sent["body"]["messages"][1] == {"role": "assistant", "content": "a"}


def test_options_are_merged_last_and_reported():
    sent: dict = {}
    provider = make_provider(
        recording_handler(openai_reply(), sent),
        options={"max_tokens": 512, "seed": 7, "temperature": 0.5},
    )

    one_turn(provider)

    body = sent["body"]
    assert body["max_tokens"] == 512
    assert body["seed"] == 7
    assert body["temperature"] == 0.5  # the override wins
    assert provider.request_options() == {"temperature": 0.5, "max_tokens": 512, "seed": 7}
    # What we log is what we sent.
    assert {k: body[k] for k in provider.request_options()} == provider.request_options()


def test_options_cannot_clobber_the_request_skeleton():
    # Settings are merged first, structure written after: an options dict with a
    # stray "model" or "messages" key cannot replace the real ones.
    sent: dict = {}
    provider = make_provider(
        recording_handler(openai_reply(), sent),
        model="real",
        options={"model": "bogus", "messages": [], "tools": "nope"},
    )

    one_turn(provider)

    assert sent["body"]["model"] == "real"
    assert sent["body"]["messages"] == [{"role": "user", "content": "hi"}]
    assert "tools" not in sent["body"]


def test_api_key_becomes_a_bearer_header_only_when_given():
    with_key: dict = {}
    one_turn(make_provider(recording_handler(openai_reply(), with_key), api_key="sk-test"))
    assert with_key["headers"]["authorization"] == "Bearer sk-test"

    without: dict = {}
    one_turn(make_provider(recording_handler(openai_reply(), without)))
    assert "authorization" not in without["headers"]


def test_no_environment_fallback_for_the_api_key(monkeypatch):
    # Deliberate: a key picked up silently would be sent to whatever base_url is
    # configured. A local server needs none; a remote one gets it explicitly.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    sent: dict = {}

    one_turn(make_provider(recording_handler(openai_reply(), sent)))

    assert "authorization" not in sent["headers"]


# --- what comes back ---------------------------------------------------------


def test_text_usage_and_finish_reason_are_parsed():
    reply = openai_reply(
        "done", finish_reason="stop", usage={"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}
    )
    msg = one_turn(make_provider(lambda req: httpx.Response(200, json=reply)))

    assert msg.role == "assistant"
    assert msg.text == "done"
    assert not msg.wants_tools
    assert msg.usage == Usage(input_tokens=120, output_tokens=30)
    assert msg.usage.total_tokens == 150
    assert msg.stop_reason == "stop"


def test_null_content_becomes_an_empty_string():
    # The real shape of a tools-only reply: "content": null. Downstream code does
    # `text.strip()` and friends, so it must be a str.
    reply = openai_reply(
        text=None,
        tool_calls=[
            {"id": "call_1", "type": "function",
             "function": {"name": "get_current_time", "arguments": '{"timezone": "UTC"}'}}
        ],
        finish_reason="tool_calls",
    )
    msg = one_turn(make_provider(lambda req: httpx.Response(200, json=reply)))

    assert msg.text == ""
    assert msg.wants_tools
    assert msg.tool_calls == [
        ToolCall(id="call_1", name="get_current_time", arguments={"timezone": "UTC"})
    ]
    assert msg.stop_reason == "tool_calls"


def test_several_tool_calls_keep_their_ids_and_order():
    reply = openai_reply(
        text=None,
        tool_calls=[
            {"id": "call_a", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a"}'}},
            {"id": "call_b", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "b"}'}},
        ],
    )
    msg = one_turn(make_provider(lambda req: httpx.Response(200, json=reply)))

    assert [c.id for c in msg.tool_calls] == ["call_a", "call_b"]
    assert [c.arguments["path"] for c in msg.tool_calls] == ["a", "b"]


def test_a_missing_tool_call_id_is_synthesized():
    # Some local servers omit ids. Without one the tool result could not be matched
    # back to its call on the next request, so we make a stable one, as Ollama does.
    reply = openai_reply(
        text=None,
        tool_calls=[{"type": "function", "function": {"name": "grep", "arguments": '{"pattern": "x"}'}}],
    )
    msg = one_turn(make_provider(lambda req: httpx.Response(200, json=reply)))

    assert msg.tool_calls == [ToolCall(id="call_0_grep", name="grep", arguments={"pattern": "x"})]


def test_arguments_already_parsed_into_a_dict_are_accepted():
    reply = openai_reply(
        text=None,
        tool_calls=[{"id": "c", "type": "function", "function": {"name": "grep", "arguments": {"pattern": "x"}}}],
    )
    msg = one_turn(make_provider(lambda req: httpx.Response(200, json=reply)))

    assert msg.tool_calls[0].arguments == {"pattern": "x"}
    assert "[cornac]" not in msg.text


def test_empty_arguments_mean_no_arguments_without_complaint():
    for raw in ("", None):
        reply = openai_reply(
            text=None,
            tool_calls=[{"id": "c", "type": "function", "function": {"name": "list_dir", "arguments": raw}}],
        )
        msg = one_turn(make_provider(lambda req, reply=reply: httpx.Response(200, json=reply)))
        assert msg.tool_calls[0].arguments == {}
        assert msg.text == ""


def test_invalid_json_arguments_become_empty_with_a_note_and_never_raise():
    # A 4B model closing a brace wrong must not kill the run. The call is kept (so
    # the wire format stays valid when the tool result is sent back), its arguments
    # are {}, and the model is told in its own text what happened.
    garbled = '{"timezone": "UTC'
    reply = openai_reply(
        text="Checking.",
        tool_calls=[{"id": "call_1", "type": "function",
                     "function": {"name": "get_current_time", "arguments": garbled}}],
    )
    msg = one_turn(make_provider(lambda req: httpx.Response(200, json=reply)))

    assert msg.tool_calls == [ToolCall(id="call_1", name="get_current_time", arguments={})]
    assert msg.text.startswith("Checking.\n")
    assert "[cornac] could not parse the arguments of tool call 'get_current_time'" in msg.text
    assert garbled in msg.text  # the raw text is shown so the slip is obvious


def test_non_object_json_arguments_are_rejected_with_a_note():
    reply = openai_reply(
        text=None,
        tool_calls=[{"id": "c", "type": "function", "function": {"name": "grep", "arguments": "[1, 2]"}}],
    )
    msg = one_turn(make_provider(lambda req: httpx.Response(200, json=reply)))

    assert msg.tool_calls[0].arguments == {}
    assert "valid JSON but not an object" in msg.text
    assert "list" in msg.text


def test_missing_and_null_usage_count_as_zero():
    # Missing keys, an absent usage block, and null values must all read as 0: the
    # agent sums usage across the run, and int + None would crash it mid-loop.
    for usage in ({}, {"prompt_tokens": None, "completion_tokens": None}):
        reply = openai_reply("bare", usage=usage)
        msg = one_turn(make_provider(lambda req, reply=reply: httpx.Response(200, json=reply)))
        assert msg.usage == Usage(input_tokens=0, output_tokens=0)
        assert Usage() + msg.usage == Usage()

    reply = openai_reply("bare")
    del reply["usage"]
    msg = one_turn(make_provider(lambda req: httpx.Response(200, json=reply)))
    assert msg.usage == Usage()


def test_a_reply_with_no_choices_is_an_empty_turn_not_a_crash():
    msg = one_turn(make_provider(lambda req: httpx.Response(200, json={})))

    assert msg.role == "assistant"
    assert msg.text == ""
    assert msg.tool_calls == []
    assert msg.stop_reason is None
    assert msg.usage == Usage()


# --- retries (the same shape as OllamaProvider) ------------------------------


def scripted_handler(statuses: list[int], calls: list[int]):
    """A handler that answers with each status in turn, recording every call."""

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        status = statuses.pop(0)
        if status == 200:
            return httpx.Response(200, json=openai_reply("ok"))
        return httpx.Response(status, text="server says no")

    return handler


def test_a_500_is_retried_once_and_then_succeeds():
    calls: list[int] = []
    sleeps: list[float] = []
    provider = make_provider(scripted_handler([500, 200], calls), backoff=0.5)
    provider._sleep = sleeps.append

    msg = one_turn(provider)

    assert msg.text == "ok"
    assert len(calls) == 2
    assert sleeps == [0.5]


def test_the_default_two_retries_give_up_after_three_attempts_with_the_last_error():
    calls: list[int] = []
    sleeps: list[float] = []
    provider = make_provider(scripted_handler([500, 503, 502, 200], calls), backoff=0.5)
    provider._sleep = sleeps.append

    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        one_turn(provider)

    assert excinfo.value.response.status_code == 502  # the *last* failure
    assert len(calls) == 3  # retries=2 means the first attempt plus two more
    assert sleeps == [0.5, 1.0]  # exponential, and no sleep after the last try


def test_retries_zero_means_exactly_one_attempt():
    calls: list[int] = []
    sleeps: list[float] = []
    provider = make_provider(scripted_handler([500, 200], calls), retries=0)
    provider._sleep = sleeps.append

    with pytest.raises(httpx.HTTPStatusError):
        one_turn(provider)

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
    assert len(calls) == 1
    assert sleeps == []


def test_a_dropped_connection_is_retried_like_a_500():
    calls: list[int] = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("connection refused")  # server still loading the model
        return httpx.Response(200, json=openai_reply("ok"))

    provider = make_provider(handler, backoff=0.25)
    provider._sleep = sleeps.append

    assert one_turn(provider).text == "ok"
    assert len(calls) == 2
    assert sleeps == [0.25]


def test_persistent_connection_failure_reraises_the_transport_error():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ConnectError("connection refused")

    with pytest.raises(httpx.ConnectError):
        one_turn(make_provider(handler, retries=2))

    assert len(calls) == 3


# --- the server's own reason survives into the error -------------------------


def test_openai_style_nested_error_message_is_kept():
    body = {"error": {"message": "The model `nope` does not exist", "type": "invalid_request_error",
                      "param": "model", "code": "model_not_found"}}
    provider = make_provider(lambda req: httpx.Response(404, json=body), base_url="http://srv:8000/v1")

    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        one_turn(provider)

    text = str(excinfo.value)
    assert "404" in text
    assert "The model `nope` does not exist" in text
    assert "http://srv:8000/v1" in text  # which server said no
    assert excinfo.value.response.status_code == 404  # .response is still there for callers


def test_vllm_style_top_level_message_is_kept():
    body = {"object": "error", "message": "This model's maximum context length is 8192 tokens",
            "type": "BadRequestError", "code": 400}
    provider = make_provider(lambda req: httpx.Response(400, json=body))

    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        one_turn(provider)

    assert "maximum context length is 8192 tokens" in str(excinfo.value)


def test_ollama_style_string_error_is_kept():
    provider = make_provider(
        lambda req: httpx.Response(500, json={"error": "cudaMalloc failed: out of memory"}), retries=0
    )

    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        one_turn(provider)

    assert "500" in str(excinfo.value)
    assert "cudaMalloc failed: out of memory" in str(excinfo.value)


def test_a_non_json_error_body_falls_back_to_its_text():
    provider = make_provider(lambda req: httpx.Response(502, text="<html>Bad Gateway</html>"), retries=0)

    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        one_turn(provider)

    assert "502" in str(excinfo.value)
    assert "Bad Gateway" in str(excinfo.value)


def test_an_empty_error_body_says_so():
    provider = make_provider(lambda req: httpx.Response(503, text=""), retries=0)

    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        one_turn(provider)

    assert "(empty response body)" in str(excinfo.value)


# --- identity and configuration ---------------------------------------------


def test_name_and_context_window():
    assert OpenAICompatibleProvider(model="qwen").name == "openai:qwen"
    # The chat endpoint does not report the model's context length, so the caller
    # says — and None leaves the agent loop's clearing disabled.
    assert OpenAICompatibleProvider().context_window is None
    assert OpenAICompatibleProvider(context_window=32768).context_window == 32768


def test_registered_in_the_providers_package():
    from cornac.providers import PROVIDERS, AnthropicProvider, OllamaProvider, Provider
    from cornac.providers import OpenAICompatibleProvider as Registered

    assert Registered is OpenAICompatibleProvider
    assert issubclass(Registered, Provider)
    assert PROVIDERS == {
        "ollama": OllamaProvider,
        "openai": OpenAICompatibleProvider,
        "anthropic": AnthropicProvider,
    }


# --- through the real agent loop (still offline) -----------------------------


def test_a_real_agent_loop_round_trips_a_tool_call_through_this_provider():
    # The wire-shape tests above check each message kind in isolation. This one
    # proves the part they cannot: that the id the server sent comes back as the
    # tool_call_id on the NEXT request, so the server can match the result to the
    # call, and that the loop then finishes on the model's final text.
    from cornac import Agent, ToolRegistry
    from cornac.tools.builtin.clock import get_current_time

    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            reply = openai_reply(
                text=None,
                tool_calls=[{"id": "call_xyz", "type": "function",
                             "function": {"name": "get_current_time",
                                          "arguments": '{"timezone": "UTC"}'}}],
                finish_reason="tool_calls",
            )
        else:
            reply = openai_reply("It is now.")
        return httpx.Response(200, json=reply)

    provider = make_provider(handler, model="m")
    agent = Agent(provider=provider, registry=ToolRegistry([get_current_time]),
                  system_prompt="Be brief.", max_steps=5)

    result = agent.run("What time is it?")

    assert result.text == "It is now."
    assert result.stop_reason == "done"
    assert result.steps == 2
    assert len(requests) == 2
    assert requests[0]["tool_choice"] == "auto"
    assistant_turn, tool_turn = requests[1]["messages"][-2:]
    assert assistant_turn["tool_calls"][0]["id"] == "call_xyz"
    assert assistant_turn["tool_calls"][0]["function"]["arguments"] == '{"timezone": "UTC"}'
    assert tool_turn["role"] == "tool"
    assert tool_turn["tool_call_id"] == "call_xyz"
    assert "UTC" in tool_turn["content"]


# --- rate limits and timeouts are transient, not "you asked wrong" ------------


def test_a_429_is_retried_and_then_succeeds():
    # OpenAI's rate limiter and a LiteLLM proxy under load answer 429; a half-second
    # pause is the right response, not the end of the run.
    calls: list[int] = []
    sleeps: list[float] = []
    provider = make_provider(scripted_handler([429, 200], calls), backoff=0.5)
    provider._sleep = sleeps.append

    assert one_turn(provider).text == "ok"
    assert len(calls) == 2
    assert sleeps == [0.5]


def test_a_408_is_retried_too():
    calls: list[int] = []
    provider = make_provider(scripted_handler([408, 200], calls), backoff=0.5)
    provider._sleep = lambda seconds: None

    assert one_turn(provider).text == "ok"
    assert len(calls) == 2


def test_persistent_429s_give_up_after_the_configured_retries():
    calls: list[int] = []
    provider = make_provider(scripted_handler([429, 429, 429, 200], calls), retries=2)
    provider._sleep = lambda seconds: None

    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        one_turn(provider)

    assert excinfo.value.response.status_code == 429
    assert len(calls) == 3


def test_other_4xx_still_raise_at_once():
    for status in (401, 403, 404, 422):
        calls: list[int] = []
        provider = make_provider(scripted_handler([status, 200], calls))
        provider._sleep = lambda seconds: None
        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            one_turn(provider)
        assert excinfo.value.response.status_code == status
        assert len(calls) == 1


def test_retry_after_is_honoured_when_longer_than_the_backoff_and_capped():
    calls: list[int] = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, text="slow down", headers={"Retry-After": "2"})
        if len(calls) == 2:
            return httpx.Response(429, text="slow down", headers={"Retry-After": "3600"})
        return httpx.Response(200, json=openai_reply("ok"))

    provider = make_provider(handler, backoff=0.5, retries=2)
    provider._sleep = sleeps.append

    assert one_turn(provider).text == "ok"
    assert sleeps == [2.0, 60.0]        # the server's ask, then the cap instead of an hour


def test_a_retry_after_shorter_than_the_backoff_or_unparseable_leaves_the_backoff():
    sleeps: list[float] = []
    statuses = [429, 429, 200]
    headers = [{"Retry-After": "0.1"}, {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}, {}]

    def handler(request: httpx.Request) -> httpx.Response:
        status, hdrs = statuses.pop(0), headers.pop(0)
        if status == 200:
            return httpx.Response(200, json=openai_reply("ok"))
        return httpx.Response(status, text="busy", headers=hdrs)

    provider = make_provider(handler, backoff=0.5, retries=2)
    provider._sleep = sleeps.append

    assert one_turn(provider).text == "ok"
    assert sleeps == [0.5, 1.0]


# --- synthesized ids are unique across the conversation -----------------------


def test_synthesized_ids_do_not_repeat_across_turns():
    # In this wire format the server matches a tool result to its call by id; two
    # calls sharing an id in one history is ambiguous to it, and to the digest.
    reply = openai_reply(
        text=None,
        tool_calls=[{"type": "function", "function": {"name": "grep", "arguments": '{"pattern": "x"}'}}],
    )
    provider = make_provider(lambda req: httpx.Response(200, json=reply))

    first = one_turn(provider).tool_calls[0].id
    second = one_turn(provider).tool_calls[0].id

    assert first == "call_0_grep"
    assert second == "call_1_grep"
    assert first != second
