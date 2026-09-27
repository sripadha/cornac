"""Anthropic (Claude) provider.

Translates the neutral conversation into Anthropic's Messages API shape and parses
the response back into a neutral assistant Message. Two Anthropic-specific quirks
are handled here so the rest of cornac never has to know about them:

  1. The system prompt is a separate top-level parameter, not a message in the list.
  2. Tool results must be delivered inside a *user* message as `tool_result` content
     blocks — so our neutral role="tool" messages get folded into user turns here.

Week 4 adds the two bits of metadata the benchmark needs — token usage and the stop
reason — which the API hands back on every response, and turns on the SDK's own
retry logic so a flaky connection does not sink a long benchmark run.

Week 4c turns on prompt caching. An agent loop re-sends the WHOLE conversation on
every step, and the front of that conversation — the system prompt (persona plus the
workspace description, easily a few thousand tokens) and the tool schemas — is
byte-for-byte identical from one step to the next, and from one run to the next. The
API will keep that prefix in a cache and charge a fraction of the input price to read
it back (a tenth, at the time of writing; a cache write costs a quarter more than a
plain read, once). On a 20-step run the prefix is paid for in full once and read
cheaply 19 times, and time-to-first-token drops too, because the model does not
re-process it. The mechanics are in `complete`; the accounting lands in
`Usage.cache_read_tokens` / `cache_write_tokens` via `_from_anthropic`.
"""

from __future__ import annotations

import os

from cornac.core.messages import Message, ToolCall, Usage
from cornac.providers.base import Provider

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# The marker that tells the API "cache the prompt up to and including this block".
# "ephemeral" is the only kind there is: a short-lived entry (minutes) that is refreshed
# every time it is hit, which is exactly the rhythm of an agent loop calling the model
# every few seconds. Copied (dict(...)) at each use so no two blocks share one object.
CACHE_CONTROL = {"type": "ephemeral"}


class AnthropicProvider(Provider):
    context_window = 200_000  # every current Claude model; the loop sizes clearing against it

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 2048, api_key: str | None = None):
        # Imported lazily so cornac is usable with only the Ollama provider installed.
        import anthropic

        self.model = model
        self.max_tokens = max_tokens
        # Unlike the Ollama provider, we don't write a retry loop here: the Anthropic
        # SDK has one built in. With max_retries=3 it re-sends on connection errors,
        # 429 rate limits and 5xx server errors, with exponential backoff between
        # attempts, and gives up after the third retry. Constructing the client does
        # not open a connection — nothing touches the network until complete() runs.
        # Tests swap this attribute for a fake whose messages.create() records its
        # arguments and returns a stub, the same way the Ollama tests swap in a
        # MockTransport — so the whole request shape is testable offline.
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            max_retries=3,
        )

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}"

    def complete(self, messages: list[Message], tools: list[dict]) -> Message:
        """Run one model turn, with the stable prefix marked for caching. See _create."""
        return self._create(messages, tools)

    def complete_without_tools(self, messages: list[Message], tools: list[dict]) -> Message:
        """One turn in which Claude may not call a tool — for the loop's wrap-up (Week 4c).

        Why this is not simply `complete(messages, [])`: the Messages API requires
        `tools` to be defined on any request whose messages contain tool_use or
        tool_result blocks — and by the time a run gives up, its history always does
        (a run cannot reach max_steps or stuck without a tool call). Omitting the
        definitions gets a 400 ("Requests which include tool_use or tool_result
        blocks must define tools"), the wrap-up fails, and the summary is None on
        every unfinished run. So the definitions are kept and their USE is forbidden
        with `tool_choice: {"type": "none"}`, which is the documented way to say
        "answer in text only" and is accepted by every current model. The tool list
        and system prompt keep their cache markers, so the cached prefix survives;
        only the messages segment (which is not marked) is re-read.
        """
        return self._create(messages, tools, tool_choice={"type": "none"})

    def _create(
        self, messages: list[Message], tools: list[dict], tool_choice: dict | None = None
    ) -> Message:
        """Build and send one request, with the stable prefix marked for caching.

        Anthropic's cache is prefix-based: the prompt is laid out as tools, then
        system, then messages, and a `cache_control` marker on a block means "cache
        everything from the start up to and including me". Two markers are placed:

          - on the LAST tool definition, so the whole tool list is one cached segment;
          - on the system block, so tools + system together are another.

        Why two and not just the system one (which already covers the tools before
        it)? Because the two change at different rates. The tool list is the same for
        every run of the benchmark; the system prompt changes whenever the workspace
        description does (a different task, a new file). With a marker on the tools
        as well, a fresh system prompt still hits the cache for the tools segment
        instead of missing on everything. The API allows four markers; two are used,
        leaving room for a caller who wants to pin a long user message too.

        Why NOT put a marker on the last message, which the docs also suggest? Two
        reasons. The loop appends to `messages` every step, so a marker there would
        have to move every call (extra bookkeeping for a marginal gain on prompts
        this size). And Week 4c's context clearing rewrites OLD tool results in place
        when the window fills, which breaks the cache prefix from that point on —
        but never touches system or tools, so the two markers used here survive a
        clear untouched. Caching the part that is guaranteed stable is the deal.

        `tool_choice`, when given, is sent alongside the tool list; complete() never
        sets it (the model decides, which is the only sane setting for an agent
        loop), complete_without_tools() sets it to "none".

        One practical note: prompts shorter than the model's minimum cacheable length
        (on the order of one to four thousand tokens, depending on the model) are
        simply not cached. The request still succeeds; `cache_read_tokens` and
        `cache_write_tokens` just stay 0. Marking is never an error.
        """
        system, api_messages = self._to_anthropic(messages)
        api_tools = [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["input_schema"],
            }
            for t in tools
        ]
        if api_tools:
            # Marker 1: the end of the tool list. New dicts are built above, so the
            # registry's own schema dicts are never mutated by this.
            api_tools[-1]["cache_control"] = dict(CACHE_CONTROL)

        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": api_messages,
        }
        if system:
            # Marker 2: the system prompt. The API takes `system` either as a plain
            # string or as a list of text blocks; only the block form can carry a
            # cache_control, hence the list-of-one.
            kwargs["system"] = [
                {"type": "text", "text": system, "cache_control": dict(CACHE_CONTROL)}
            ]
        if api_tools:
            kwargs["tools"] = api_tools
            # Only meaningful next to a tool list: with no tools there is nothing to
            # forbid, and the API rejects a tool_choice that has no tools to apply to.
            if tool_choice is not None:
                kwargs["tool_choice"] = dict(tool_choice)

        response = self._client.messages.create(**kwargs)
        return self._from_anthropic(response)

    # --- neutral -> Anthropic ------------------------------------------------

    def _to_anthropic(self, messages: list[Message]) -> tuple[str, list[dict]]:
        """Split the neutral list into (system text, API messages).

        The system text comes back as a plain string; `complete` wraps it into a
        cache-marked block. Keeping the translation and the caching policy apart
        means this function stays a pure reshaping that tests can check directly.
        """
        system_parts: list[str] = []
        api_messages: list[dict] = []

        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.text)

            elif msg.role == "user":
                api_messages.append({"role": "user", "content": msg.text})

            elif msg.role == "assistant":
                content: list[dict] = []
                # The API rejects a text block that is empty or whitespace-only, so
                # only real text becomes a block.
                if msg.text and msg.text.strip():
                    content.append({"type": "text", "text": msg.text})
                for call in msg.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.arguments,
                        }
                    )
                # The API also rejects an assistant message with no content at all
                # ({"role": "assistant", "content": []}). That shape is reachable: a
                # reply of "" or one whose only blocks were of a kind _from_anthropic
                # ignores. It bites on the NEXT complete() — when the whole history
                # is re-sent — so an Agent reused for a second run() would fail. An
                # empty turn carries no information, so it is simply left out.
                # (Consecutive same-role turns are fine: the API merges them.)
                if content:
                    api_messages.append({"role": "assistant", "content": content})

            elif msg.role == "tool":
                # Anthropic wants tool results inside a user message. Merge into the
                # previous user message if it's already a tool-result turn, so that
                # all results for one assistant turn arrive together.
                block = {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.text,
                }
                if msg.is_error:
                    block["is_error"] = True

                if (
                    api_messages
                    and api_messages[-1]["role"] == "user"
                    and isinstance(api_messages[-1]["content"], list)
                ):
                    api_messages[-1]["content"].append(block)
                else:
                    api_messages.append({"role": "user", "content": [block]})

        return "\n\n".join(system_parts), api_messages

    # --- Anthropic -> neutral ------------------------------------------------

    def _from_anthropic(self, response) -> Message:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))

        # A real API response always carries `usage`, but a hand-built one in a test
        # (or a future response type) may not — so read it defensively, and leave
        # Message.usage as None when there is genuinely nothing to report.
        usage = None
        api_usage = getattr(response, "usage", None)
        if api_usage is not None:
            # With caching on, `input_tokens` is only the UNCACHED part of the prompt;
            # what was read from the cache and what was written into it come back in
            # two separate counters. All three are kept apart in Usage — the
            # benchmark wants to see the cache working, and Usage.prompt_tokens adds
            # them back up for anyone who just wants "how big was the prompt". The
            # cache fields are None on the SDK's Usage when caching was not involved
            # (and absent on older response shapes), so both read as 0 — which keeps
            # Usage(...) equal to what this returned before caching existed.
            usage = Usage(
                input_tokens=getattr(api_usage, "input_tokens", 0) or 0,
                output_tokens=getattr(api_usage, "output_tokens", 0) or 0,
                cache_read_tokens=getattr(api_usage, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(api_usage, "cache_creation_input_tokens", 0) or 0,
            )

        return Message(
            role="assistant",
            text="".join(text_parts),
            tool_calls=tool_calls,
            usage=usage,
            # Anthropic's own word for why it stopped: "end_turn", "tool_use", or
            # "max_tokens" (cut off — worth knowing when a run looks truncated).
            stop_reason=getattr(response, "stop_reason", None),
        )
