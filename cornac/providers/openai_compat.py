"""OpenAI-compatible (local inference server) provider.

The second local provider, and a sibling of OllamaProvider: read the two side by side.
OllamaProvider speaks Ollama's own /api/chat; this one speaks the OpenAI Chat
Completions wire format, which has become the lingua franca of inference servers.
vLLM, llama.cpp's llama-server, LM Studio, SGLang, TGI, LiteLLM proxies — and OpenAI
itself — all accept `POST {base_url}/chat/completions` with the same JSON, so one small
file lets cornac drive any of them, with zero changes to the agent, tools or permissions.

Why a second local provider at all? Ollama is one runtime with one engine. Once you
care about inference itself — how a KV cache is managed, how requests are batched, how
a quantized model behaves — you want the SAME model on vLLM and on llama.cpp and a
fair comparison between them. The Provider seam is what makes that comparison cost
one file rather than a fork of the harness.

What differs from Ollama's format, all absorbed here:

  1. Tool-call arguments arrive as a JSON *string*, not a parsed dict, and go back the
     same way. A local model can garble that string, so parsing must never raise.
  2. Tool results are matched to their calls by `tool_call_id`, not by name/order, so
     every call needs an id — we synthesize one when a server leaves it out.
  3. Generation settings (temperature, max_tokens, ...) sit at the top level of the
     request, not under an "options" key.
  4. Token counts live in `usage.prompt_tokens` / `usage.completion_tokens`, and the
     stop reason is `choices[0].finish_reason`.
  5. Error bodies vary by server: OpenAI nests {"error": {"message": ...}}, vLLM puts
     "message" at the top level, Ollama's compatibility endpoint sends a plain string.
     `_with_server_reason` reads all three so a failure always says why.

Retries, backoff and the "keep the server's own reason" error handling are copied
from OllamaProvider on purpose. The shared shape is the lesson; the wire format is
the only thing that changes between two local backends.
"""

from __future__ import annotations

import json
import time

import httpx

from cornac.core.messages import Message, ToolCall, Usage
from cornac.providers.base import Provider

# vLLM's default port. llama-server listens on 8080 and LM Studio on 1234 by default;
# pass base_url to point at them. The "/v1" is part of the OpenAI convention.
DEFAULT_BASE_URL = "http://localhost:8000/v1"

# The two 4xx statuses that mean "not now" rather than "not like that": 429 Too Many
# Requests is how OpenAI's rate limiter and a LiteLLM proxy under load say it, 408 is
# a request timeout. Resending the same bytes after a pause is exactly the right move
# for both — unlike every other 4xx, where it cannot help. Ollama never sends either,
# which is why its sibling provider does not bother.
RETRIABLE_CLIENT_ERRORS = (408, 429)
# A Retry-After header is honoured, but a server asking for an hour does not get to
# hang an agent run: wait this long at most.
MAX_RETRY_AFTER = 60.0


class OpenAICompatibleProvider(Provider):
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = "",
        api_key: str | None = None,
        timeout: float = 120.0,
        options: dict | None = None,
        context_window: int | None = None,
        retries: int = 2,
        backoff: float = 0.5,
    ):
        """Configure a connection to an OpenAI-compatible chat server.

        Args:
            base_url: the server's API root, including the "/v1" most servers mount
                under. A trailing slash is tolerated.
            model: the model name to request. Multi-model servers (vLLM, OpenAI) need
                it; single-model ones (llama-server) ignore it, so "" is a workable
                default there. It is always sent, because the spec says it is
                required and a server that cares will answer with a clear error.
            api_key: sent as `Authorization: Bearer <key>` ONLY when given. There is
                deliberately no fallback to an environment variable: the default target
                is a local server that needs no key, and a key picked up silently from
                the environment would be sent to whatever base_url happens to be
                configured — a leak waiting to happen. Be explicit.
            timeout: seconds to wait for one response. Shorter than Ollama's 300 s
                because these servers are usually run for speed, but still generous
                for a long prompt on a laptop GPU.
            options: extra top-level request fields (max_tokens, seed, top_p, ...),
                merged LAST so they can override the defaults in request_options().
                Kept to a minimum by default because servers differ in which fields
                they accept, and a strict one rejects unknown fields outright.
            context_window: the model's context length in tokens, if known. Unlike
                Ollama, the chat endpoint does not tell us, so the caller says. None
                disables the agent loop's context clearing.
            retries: how many times to re-send after the FIRST attempt fails — the
                same meaning as OllamaProvider.max_retries and the Anthropic SDK's
                max_retries. 0 makes exactly one attempt; the default 2 allows three.
            backoff: seconds to wait before the first retry; doubles each time.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.options = dict(options or {})
        self.context_window = context_window
        self.retries = retries
        self.backoff = backoff
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(timeout=timeout, headers=headers)
        # Kept as an attribute so tests can swap it for a no-op and not actually wait.
        self._sleep = time.sleep
        # How many tool-call ids this provider has made up so far (see _from_openai).
        self._synthesized_ids = 0

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def request_options(self) -> dict:
        """The generation settings merged into every request.

        Only temperature 0 by default — the one field every compatible server accepts
        and the one that matters most for a reproducible benchmark. `options` is
        merged last so a caller can add a seed or max_tokens for a server that takes
        them, or override the temperature. Logged per run by the benchmark: a result
        nobody can reproduce is not a result.
        """
        options = {"temperature": 0.0}
        options.update(self.options)
        return options

    def complete(self, messages: list[Message], tools: list[dict]) -> Message:
        # Settings first, structure second: model/messages/tools are written after
        # the merge so an `options` dict can never clobber the request's skeleton —
        # and the skeleton's keys are popped from the settings so a stray "tools" in
        # options cannot slip through on a call that sends none.
        payload: dict = dict(self.request_options())
        for reserved in ("model", "messages", "tools", "tool_choice"):
            payload.pop(reserved, None)
        payload["model"] = self.model
        payload["messages"] = self._to_openai(messages)
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["input_schema"],
                    },
                }
                for t in tools
            ]
            # "auto" = the model decides whether to call a tool or answer in text —
            # the only sane setting for an agent loop. Sent only alongside tools:
            # several servers reject tool_choice when no tools are given.
            payload["tool_choice"] = "auto"

        data = self._post_with_retries(f"{self.base_url}/chat/completions", payload)
        return self._from_openai(data)

    # --- HTTP with retries ---------------------------------------------------

    def _post_with_retries(self, url: str, payload: dict) -> dict:
        """POST `payload` to `url` and return the parsed JSON, retrying transient failures.

        Same reasoning as OllamaProvider._post_with_retries: a local server is one
        process that can be caught mid model-load, and a dropped connection
        (httpx.TransportError) or a 5xx usually clears within a second. Most 4xx mean
        the request itself is wrong (unknown model, a field the server rejects) and
        resending the same bytes cannot help, so they are raised at once — with the
        server's own explanation attached. The exceptions are 429 and 408
        (RETRIABLE_CLIENT_ERRORS): "too many requests" and "request timeout" are the
        canonical TRANSIENT failures of the servers this provider targets — OpenAI
        itself, a LiteLLM proxy, a busy vLLM — and a rate limit that ends a
        twenty-step run at step twelve, when a half-second pause would have carried
        it, is a wasted run. Those are retried like a 5xx.

        Waits are backoff, 2*backoff, 4*backoff, ... — doubling so a server that is
        genuinely busy gets progressively more breathing room — or the server's own
        Retry-After when it names one and it is longer (capped at MAX_RETRY_AFTER).
        """
        attempts = 1 + max(0, self.retries)
        for attempt in range(attempts):
            retry_after: float | None = None
            try:
                resp = self._client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status < 500 and status not in RETRIABLE_CLIENT_ERRORS:
                    raise self._with_server_reason(e) from e
                error: Exception = self._with_server_reason(e)
                retry_after = _retry_after_seconds(e.response)
            except httpx.TransportError as e:
                error = e

            # Don't sleep after the final attempt — there is nothing left to wait for.
            if attempt < attempts - 1:
                delay = self.backoff * 2**attempt
                if retry_after is not None:
                    delay = max(delay, min(retry_after, MAX_RETRY_AFTER))
                self._sleep(delay)

        # Every attempt failed: surface the last error rather than a vague one.
        raise error

    def _with_server_reason(self, e: httpx.HTTPStatusError) -> httpx.HTTPStatusError:
        """Rebuild an HTTP error so its message carries the server's own explanation.

        httpx's message is only the status line. The real diagnosis is in the body,
        but "OpenAI-compatible" stops at the happy path — error bodies differ:

            OpenAI / llama-server:  {"error": {"message": "...", "type": ..., "code": ...}}
            vLLM:                   {"object": "error", "message": "...", "type": ..., "code": ...}
            Ollama's /v1 endpoint:  {"error": "..."}

        All three are read, then a non-JSON body (a proxy's HTML page) falls back to
        its text. The base_url goes in the message too: with several servers on one
        machine, "which one said no?" is the first question. The original request and
        response stay attached so callers can still read .response.status_code.
        """
        resp = e.response
        reason = None
        try:
            body = resp.json()
        except ValueError:  # the body was not JSON
            body = None
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                reason = err.get("message")
            elif isinstance(err, str):
                reason = err
            if not reason:
                reason = body.get("message")
        if not reason:
            reason = resp.text.strip()[:300] or "(empty response body)"
        return httpx.HTTPStatusError(
            f"Server at {self.base_url} returned {resp.status_code}: {reason}",
            request=e.request,
            response=resp,
        )

    # --- neutral -> OpenAI ---------------------------------------------------

    def _to_openai(self, messages: list[Message]) -> list[dict]:
        out: list[dict] = []
        for msg in messages:
            if msg.role in ("system", "user"):
                out.append({"role": msg.role, "content": msg.text})

            elif msg.role == "assistant":
                # An empty string rather than null for a tools-only turn: OpenAI
                # itself returns null there and accepts either back, but some local
                # servers' chat templates choke on null. "" works everywhere.
                m: dict = {"role": "assistant", "content": msg.text or ""}
                if msg.tool_calls:
                    m["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            # Arguments travel as a JSON STRING in this format — the
                            # model wrote them as text and the server echoes that.
                            "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                        }
                        for c in msg.tool_calls
                    ]
                out.append(m)

            elif msg.role == "tool":
                # Matched to its call by id, not by name/order as Ollama does.
                out.append({"role": "tool", "tool_call_id": msg.tool_call_id, "content": msg.text})

        return out

    # --- OpenAI -> neutral ---------------------------------------------------

    def _from_openai(self, data: dict) -> Message:
        # A well-formed reply has exactly one choice (we never ask for more). Read it
        # defensively all the same: a bare {} from a misbehaving server should come
        # back as an empty assistant turn the loop can nudge, not a KeyError that
        # kills the run.
        choices = data.get("choices") or []
        first = choices[0] if choices else {}
        message = first.get("message") or {}
        # `content` is null (not "") when the reply is only tool calls.
        text = message.get("content") or ""

        tool_calls: list[ToolCall] = []
        notes: list[str] = []
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            name = fn.get("name") or ""
            # OpenAI always sends an id; some local servers do not. Synthesize one
            # the same way OllamaProvider does — numbered across the conversation,
            # because in THIS wire format the server matches a tool result to its
            # call by id, and a history in which two different calls share an id is
            # ambiguous to it (and to the digest, which pairs results the same way).
            call_id = call.get("id") or self._next_call_id(name)
            arguments, note = self._parse_arguments(name, fn.get("arguments"))
            if note:
                notes.append(note)
            tool_calls.append(ToolCall(id=call_id, name=name, arguments=arguments))

        if notes:
            # The note goes into the assistant's own text so the model reads it in
            # its history on the next turn, next to the tool's error about the
            # empty arguments — two signals pointing at the same slip.
            text = "\n".join(([text] if text else []) + notes)

        # OpenAI-style servers fold any cached prompt tokens INSIDE prompt_tokens
        # (some report the cached part again under prompt_tokens_details). Only the
        # two totals are read; mapping the cached part into Usage.cache_read_tokens
        # as well would count it twice in Usage.prompt_tokens. The `or 0` matters:
        # the agent sums usage across the run, and int + None would crash it.
        usage_data = data.get("usage") or {}
        usage = Usage(
            input_tokens=usage_data.get("prompt_tokens") or 0,
            output_tokens=usage_data.get("completion_tokens") or 0,
        )

        return Message(
            role="assistant",
            text=text,
            tool_calls=tool_calls,
            usage=usage,
            # The server's own word for why it stopped: "stop", "tool_calls", or
            # "length" (cut off by max_tokens / the context limit). Kept verbatim.
            stop_reason=first.get("finish_reason"),
        )

    def _next_call_id(self, name: str) -> str:
        """A tool-call id unique for the life of this provider: call_<n>_<name>."""
        call_id = f"call_{self._synthesized_ids}_{name or 'tool'}"
        self._synthesized_ids += 1
        return call_id

    @staticmethod
    def _parse_arguments(name: str, raw) -> tuple[dict, str | None]:
        """Turn a tool call's raw `arguments` into a dict — never raising.

        Returns (arguments, note): `note` is None on success, otherwise a one-line
        explanation for the model.

        The wire format says arguments are a JSON string, and a 4B model closing a
        brace wrong is a Tuesday, not an emergency. Raising here would kill the whole
        run() over one garbled call, when the loop already has a path for bad tool
        calls: the tool is invoked with {} , its validation fails, the model reads the
        error and tries again. Keeping the call (rather than dropping it) also keeps
        the wire format valid — an assistant tool_call with no matching tool result
        on the next request is rejected by every server. A few servers return the
        arguments already parsed into a dict; that is accepted as-is.
        """
        if raw is None or raw == "":
            return {}, None  # a tool with no parameters, or the model sent none
        if isinstance(raw, dict):
            return dict(raw), None
        if not isinstance(raw, str):
            return {}, (
                f"[cornac] the arguments of tool call {name!r} were not a JSON string "
                f"(got {type(raw).__name__}); the call ran with no arguments."
            )
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            return {}, (
                f"[cornac] could not parse the arguments of tool call {name!r} as JSON "
                f"({e.msg} at position {e.pos}); the call ran with no arguments. "
                f"Raw arguments: {raw[:200]!r}"
            )
        if not isinstance(parsed, dict):
            return {}, (
                f"[cornac] the arguments of tool call {name!r} were valid JSON but not an "
                f"object (got {type(parsed).__name__}); the call ran with no arguments."
            )
        return parsed, None


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """The Retry-After header as seconds, or None if absent or not a plain number.

    The header may also be an HTTP date; that form is rare on these servers and is
    ignored rather than parsed — the backoff schedule then stands on its own.
    """
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if seconds >= 0 else None
