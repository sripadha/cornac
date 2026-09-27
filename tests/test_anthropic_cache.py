"""Offline tests for Anthropic prompt caching (Week 4c) — no network, no API key.

The SDK client is swapped for a fake whose `messages.create(**kwargs)` records the
keyword arguments and returns a stub response — the same trick the Ollama tests play
with httpx.MockTransport, one layer up. That lets a test see exactly what would have
gone on the wire (which blocks carry cache_control, in what shape) and feed back any
usage shape it likes, including the cache counters a real response carries.

The stub response uses types.SimpleNamespace in place of the SDK's pydantic models:
`_from_anthropic` reads attributes, and a namespace has attributes.
"""

import types

from cornac.core.messages import Message, ToolCall, ToolResult, Usage
from cornac.providers.anthropic import CACHE_CONTROL, AnthropicProvider

# --- helpers -----------------------------------------------------------------


class FakeMessages:
    """Stands in for `client.messages`: records every create() call, answers the stub."""

    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response):
        self.messages = FakeMessages(response)


def stub_response(text: str = "ok", usage=None, stop_reason: str = "end_turn", content=None):
    """A response object shaped like the SDK's Message."""
    if content is None:
        content = [types.SimpleNamespace(type="text", text=text)]
    if usage is None:
        usage = types.SimpleNamespace(input_tokens=10, output_tokens=5)
    return types.SimpleNamespace(content=content, usage=usage, stop_reason=stop_reason)


def make_provider(response=None, **kwargs) -> tuple[AnthropicProvider, FakeMessages]:
    """An AnthropicProvider whose SDK client is a recording fake."""
    provider = AnthropicProvider(api_key="dummy-key-no-network", **kwargs)
    client = FakeClient(response or stub_response())
    provider._client = client
    return provider, client.messages


def sent(fake: FakeMessages) -> dict:
    assert len(fake.calls) == 1, "expected exactly one messages.create() call"
    return fake.calls[0]


SCHEMA_A = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
SCHEMA_B = {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}
TOOLS = [
    {"name": "read_file", "description": "Read a file.", "input_schema": SCHEMA_A},
    {"name": "grep", "description": "Search files.", "input_schema": SCHEMA_B},
]


# --- what goes on the wire: the two cache markers ----------------------------


def test_system_prompt_is_sent_as_one_cache_marked_text_block():
    # The API only accepts cache_control on the block form of `system`, hence a
    # list of one block instead of the plain string it also allows.
    provider, fake = make_provider()

    provider.complete([Message.system("Be brief."), Message.user("hi")], tools=[])

    assert sent(fake)["system"] == [
        {"type": "text", "text": "Be brief.", "cache_control": {"type": "ephemeral"}}
    ]


def test_several_system_messages_are_joined_into_the_one_block():
    provider, fake = make_provider()

    provider.complete(
        [Message.system("Persona."), Message.system("Workspace."), Message.user("hi")], tools=[]
    )

    system = sent(fake)["system"]
    assert len(system) == 1
    assert system[0]["text"] == "Persona.\n\nWorkspace."
    assert system[0]["cache_control"] == CACHE_CONTROL


def test_only_the_last_tool_carries_the_cache_marker():
    # A marker means "cache everything up to and including me", so one marker on the
    # LAST tool caches the whole list. Markers on every tool would burn the four the
    # API allows for no extra benefit.
    provider, fake = make_provider()

    provider.complete([Message.user("hi")], tools=TOOLS)

    tools = sent(fake)["tools"]
    assert [t["name"] for t in tools] == ["read_file", "grep"]
    assert "cache_control" not in tools[0]
    assert tools[1]["cache_control"] == {"type": "ephemeral"}
    # The rest of the last tool is intact, not replaced by the marker.
    assert tools[1]["description"] == "Search files."
    assert tools[1]["input_schema"] == SCHEMA_B


def test_the_registry_schemas_are_not_mutated_by_marking():
    # `tools` is the ToolRegistry's own list, reused on every step and by every
    # provider. Marking must happen on copies or the marker would leak into the
    # Ollama wire format on a mixed-provider run.
    provider, fake = make_provider()
    original = [dict(t) for t in TOOLS]

    provider.complete([Message.user("hi")], tools=TOOLS)

    assert TOOLS == original
    assert all("cache_control" not in t for t in TOOLS)


def test_a_single_tool_is_both_first_and_last_so_it_is_marked():
    provider, fake = make_provider()

    provider.complete([Message.user("hi")], tools=TOOLS[:1])

    tools = sent(fake)["tools"]
    assert len(tools) == 1
    assert tools[0]["cache_control"] == {"type": "ephemeral"}


def test_no_system_and_no_tools_sends_neither_key():
    # Nothing to cache, nothing to mark: the request must not grow an empty
    # `system: []` or `tools: []` that the API would reject.
    provider, fake = make_provider()

    provider.complete([Message.user("hi")], tools=[])

    kwargs = sent(fake)
    assert "system" not in kwargs
    assert "tools" not in kwargs
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_model_and_max_tokens_are_forwarded():
    provider, fake = make_provider(model="claude-test-1", max_tokens=321)

    provider.complete([Message.user("hi")], tools=[])

    kwargs = sent(fake)
    assert kwargs["model"] == "claude-test-1"
    assert kwargs["max_tokens"] == 321


def test_the_cache_markers_are_distinct_objects():
    # dict(CACHE_CONTROL) at each use: two blocks must never share one dict, or a
    # later in-place tweak to one marker would silently change the other.
    provider, fake = make_provider()

    provider.complete([Message.system("s"), Message.user("hi")], tools=TOOLS)

    kwargs = sent(fake)
    system_marker = kwargs["system"][0]["cache_control"]
    tool_marker = kwargs["tools"][-1]["cache_control"]
    assert system_marker == tool_marker
    assert system_marker is not tool_marker
    assert system_marker is not CACHE_CONTROL


# --- end to end through complete(): the conversation still folds correctly ---


def test_complete_round_trips_a_tool_conversation():
    call = ToolCall(id="toolu_1", name="read_file", arguments={"path": "a.py"})
    messages = [
        Message.system("Be brief."),
        Message.user("Read a.py"),
        Message(role="assistant", text="", tool_calls=[call]),
        Message.from_tool_result(ToolResult(tool_call_id="toolu_1", name="read_file", content="print(1)")),
    ]
    reply = stub_response(text="It prints 1.", usage=types.SimpleNamespace(input_tokens=80, output_tokens=6))
    provider, fake = make_provider(reply)

    msg = provider.complete(messages, TOOLS)

    kwargs = sent(fake)
    assert kwargs["messages"] == [
        {"role": "user", "content": "Read a.py"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a.py"}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "print(1)"}],
        },
    ]
    assert msg.text == "It prints 1."
    assert not msg.wants_tools
    assert msg.stop_reason == "end_turn"
    assert msg.usage == Usage(input_tokens=80, output_tokens=6)


# --- what comes back: the cache counters -------------------------------------


def test_cache_read_and_write_counters_are_mapped_into_usage():
    # The steady state of an agent loop: the prefix was read from the cache, only
    # the new tail was billed as plain input.
    usage = types.SimpleNamespace(
        input_tokens=100,
        output_tokens=20,
        cache_read_input_tokens=900,
        cache_creation_input_tokens=0,
    )
    provider, _ = make_provider(stub_response(usage=usage))

    msg = provider.complete([Message.user("hi")], tools=[])

    assert msg.usage == Usage(
        input_tokens=100, output_tokens=20, cache_read_tokens=900, cache_write_tokens=0
    )
    # The prompt was 1000 tokens long whatever it was billed as: that is the number
    # the loop compares with the context window.
    assert msg.usage.prompt_tokens == 1000
    assert msg.usage.total_tokens == 1020


def test_the_first_call_of_a_run_shows_up_as_a_cache_write():
    usage = types.SimpleNamespace(
        input_tokens=50,
        output_tokens=10,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=950,
    )
    provider, _ = make_provider(stub_response(usage=usage))

    msg = provider.complete([Message.user("hi")], tools=[])

    assert msg.usage.cache_write_tokens == 950
    assert msg.usage.cache_read_tokens == 0
    assert msg.usage.prompt_tokens == 1000


def test_a_response_without_cache_fields_behaves_exactly_as_before():
    # Older response shapes (and the existing tests' stubs) carry only the two
    # classic counters. They must parse to the same Usage they always did, which
    # dataclass equality checks field by field — cache fields included, as 0.
    usage = types.SimpleNamespace(input_tokens=50, output_tokens=12)
    provider, _ = make_provider(stub_response(usage=usage))

    msg = provider.complete([Message.user("hi")], tools=[])

    assert msg.usage == Usage(input_tokens=50, output_tokens=12)
    assert msg.usage.cache_read_tokens == 0
    assert msg.usage.cache_write_tokens == 0
    assert msg.usage.prompt_tokens == 50


def test_null_cache_fields_count_as_zero():
    # The SDK's Usage has the cache fields as Optional[int]: None when caching was
    # not involved. int + None would crash the agent's running total.
    usage = types.SimpleNamespace(
        input_tokens=50,
        output_tokens=12,
        cache_read_input_tokens=None,
        cache_creation_input_tokens=None,
    )
    provider, _ = make_provider(stub_response(usage=usage))

    msg = provider.complete([Message.user("hi")], tools=[])

    assert msg.usage == Usage(input_tokens=50, output_tokens=12)
    assert Usage() + msg.usage == msg.usage  # exactly the sum the agent loop performs


def test_no_usage_at_all_is_still_none():
    # "Unknown" stays unknown — caching adds counters, it does not invent a zero
    # where the response said nothing.
    provider, _ = make_provider(types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text="hi")]))

    msg = provider.complete([Message.user("hi")], tools=[])

    assert msg.usage is None


def test_cache_counters_add_up_across_a_run():
    # What the benchmark reports: step 1 writes the prefix, steps 2..n read it. The
    # per-run Usage must keep the cache columns, not fold them into input_tokens.
    first = Usage(input_tokens=50, output_tokens=10, cache_write_tokens=950)
    later = Usage(input_tokens=100, output_tokens=20, cache_read_tokens=950)

    total = first + later + later

    assert total == Usage(
        input_tokens=250, output_tokens=50, cache_read_tokens=1900, cache_write_tokens=950
    )
    assert total.prompt_tokens == 3100


# --- through the real SDK client (still offline) -----------------------------


def test_the_real_sdk_client_sends_the_markers_and_parses_the_cache_counters():
    # The fakes above assume the SDK forwards our dicts verbatim and exposes the two
    # cache counters as attributes. This pins both against the installed SDK: a real
    # anthropic.Anthropic client, with its HTTP routed to a MockTransport, so the
    # request body is exactly what the API would receive and the response goes
    # through the SDK's own pydantic Usage model.
    import json

    import anthropic
    import httpx

    sent: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test",
                "content": [{"type": "text", "text": "hi back"}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 7, "output_tokens": 3,
                          "cache_read_input_tokens": 900, "cache_creation_input_tokens": 0},
            },
        )

    provider = AnthropicProvider(api_key="dummy-key-no-network")
    provider._client = anthropic.Anthropic(
        api_key="dummy-key-no-network",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    msg = provider.complete([Message.system("sys"), Message.user("hi")], TOOLS)

    body = sent["body"]
    assert body["system"] == [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}]
    assert [(t["name"], t.get("cache_control")) for t in body["tools"]] == [
        ("read_file", None),
        ("grep", {"type": "ephemeral"}),
    ]
    assert msg.text == "hi back"
    assert msg.usage == Usage(input_tokens=7, output_tokens=3, cache_read_tokens=900, cache_write_tokens=0)
    assert msg.usage.prompt_tokens == 907


# --- tools off, the Anthropic way: definitions kept, use forbidden -----------
#
# The loop's wrap-up asks for a text-only turn (Provider.complete_without_tools).
# On this API "no tools" cannot be said by sending none: a history that contains
# tool_use/tool_result blocks is rejected with 400 unless `tools` is defined. So the
# definitions are still sent and tool_choice "none" forbids their use. These pin
# both halves — and that the ordinary complete() never grows a tool_choice.

TOOL_HISTORY = [
    Message.system("Be brief."),
    Message.user("Read a.py"),
    Message(role="assistant", text="", tool_calls=[ToolCall("toolu_1", "read_file", {"path": "a.py"})]),
    Message.from_tool_result(ToolResult(tool_call_id="toolu_1", name="read_file", content="print(1)")),
]


def test_complete_never_sends_tool_choice():
    provider, fake = make_provider()

    provider.complete(TOOL_HISTORY, TOOLS)

    assert "tool_choice" not in sent(fake)


def test_complete_without_tools_keeps_the_definitions_and_forbids_their_use():
    provider, fake = make_provider(stub_response(text="Unfinished: I read a.py."))

    msg = provider.complete_without_tools(TOOL_HISTORY, TOOLS)

    kwargs = sent(fake)
    assert [t["name"] for t in kwargs["tools"]] == ["read_file", "grep"]   # still defined
    assert kwargs["tool_choice"] == {"type": "none"}                       # but off limits
    assert kwargs["tools"][-1]["cache_control"] == {"type": "ephemeral"}   # cache marker intact
    # The history with its tool blocks went out unchanged.
    assert kwargs["messages"][1]["content"][0]["type"] == "tool_use"
    assert kwargs["messages"][2]["content"][0]["type"] == "tool_result"
    assert msg.text == "Unfinished: I read a.py."


def test_complete_without_tools_and_no_tools_sends_neither_key():
    # Nothing to forbid: no `tools`, and no `tool_choice` either, which the API
    # rejects when it has no tool list to apply to.
    provider, fake = make_provider()

    provider.complete_without_tools([Message.user("hi")], [])

    kwargs = sent(fake)
    assert "tools" not in kwargs and "tool_choice" not in kwargs


def test_the_wrap_up_request_through_the_real_sdk_carries_tools_and_tool_choice_none():
    # The finding's reproduction, kept as a test: an Agent on this provider runs out
    # of steps after a tool call, and its wrap-up request — the very bytes the API
    # would receive — must define the tools and forbid them, or the API answers 400
    # and every unfinished Anthropic run has summary None.
    import json

    import anthropic
    import httpx

    from cornac import Agent, ToolRegistry
    from cornac.tools.base import tool

    @tool()
    def mark(label: str = "x") -> str:
        """Record that this tool ran."""
        return f"marked {label}"

    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if len(bodies) == 1:
            content = [{"type": "tool_use", "id": "toolu_1", "name": "mark", "input": {"label": "a"}}]
            stop = "tool_use"
        else:
            if "tools" not in body or body.get("tool_choice") != {"type": "none"}:
                # What the real API does with a tool-bearing history and no tools.
                return httpx.Response(400, json={"type": "error", "error": {
                    "type": "invalid_request_error",
                    "message": "Requests which include `tool_use` or `tool_result` blocks must define tools.",
                }})
            content = [{"type": "text", "text": "Unfinished: marked a; nothing verified."}]
            stop = "end_turn"
        return httpx.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test",
            "content": content, "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": 7, "output_tokens": 3},
        })

    provider = AnthropicProvider(api_key="dummy-key-no-network")
    provider._client = anthropic.Anthropic(
        api_key="dummy-key-no-network", max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    agent = Agent(provider=provider, registry=ToolRegistry([mark]), system_prompt="sys", max_steps=1)

    result = agent.run("go")

    assert result.stop_reason == "max_steps"
    assert result.summary == "Unfinished: marked a; nothing verified."
    wrap_up = bodies[1]
    assert [t["name"] for t in wrap_up["tools"]] == ["mark"]
    assert wrap_up["tool_choice"] == {"type": "none"}
    block_types = {b["type"] for m in wrap_up["messages"] if isinstance(m["content"], list) for b in m["content"]}
    assert {"tool_use", "tool_result"} <= block_types
    assert wrap_up["messages"][-1]["content"].startswith("The run was stopped: you used all 1 of your steps")
