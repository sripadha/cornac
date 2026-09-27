"""Ollama (local model) provider.

Talks to a local Ollama daemon's /api/chat endpoint over HTTP. Ollama uses the
OpenAI-flavored message shape, which differs from Anthropic's in a few ways that we
absorb here so the agent loop stays identical across providers:

  1. The system prompt is just a message with role="system" in the list (no hoisting).
  2. Tool results are their own messages with role="tool" (no folding into user turns).
  3. Tools are wrapped as {"type": "function", "function": {...}} and the argument
     schema is called "parameters", not "input_schema".
  4. Ollama does not always assign ids to tool calls, so we synthesize stable ones.

The fact that supporting a local 7B model takes one small file like this — and zero
changes to the agent, tools, or permissions — is the whole point of the Provider seam.

Week 4 hardens this file for the benchmark, which will hammer a local daemon with
hundreds of runs and needs each one to be comparable and to survive hiccups:

  - the context window is set explicitly (Ollama's default silently truncates),
  - generation is pinned with temperature 0 AND a seed (temperature 0 alone is not
    determinism),
  - transient failures — dropped connections and 5xx — are retried with backoff,
  - token counts and the stop reason are read off the response into Message.usage
    and Message.stop_reason,
  - when the daemon refuses a request, its own explanation ("cudaMalloc failed: out
    of memory", "model not found") is kept in the raised error instead of a bare
    status code — a failure should say why (see _with_daemon_reason).
"""

from __future__ import annotations

import time

import httpx

from cornac.core.messages import Message, ToolCall, Usage
from cornac.providers.base import Provider

# The benchmark model, FROZEN on 2026-09-22 after a three-round spike (benchmark/spike/):
# on three coding tasks x 3 runs, qwen3.5:4b went 9/9 at ~20 s/run entirely on a 6 GB GPU
# once presence_penalty was forced to 0 (see below) — matching the Unsloth Q6_K/Q8_0
# builds of the same weights at 2.5-5x their speed. Change this only with new evidence.
DEFAULT_MODEL = "qwen3.5:4b"
DEFAULT_HOST = "http://localhost:11434"


class OllamaProvider(Provider):
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: float = 300.0,
        num_ctx: int = 8192,
        seed: int = 42,
        temperature: float = 0.0,
        max_retries: int = 3,
        backoff: float = 0.5,
        num_predict: int = 2048,
        presence_penalty: float = 0.0,
        think: bool | None = None,
        extra_options: dict | None = None,
    ):
        """Configure a connection to a local Ollama daemon.

        Args:
            model: the Ollama model tag to chat with.
            host: where the daemon listens.
            timeout: seconds to wait for one response. Generous, because a 7B model on
                a laptop can legitimately take a minute or two on a long prompt.
            num_ctx: the context window, in tokens. Ollama's default is 2048, and when
                the prompt is longer than that it SILENTLY truncates the prompt to fit —
                there is no error, no flag in the response, only a warning in the
                daemon's own log. An agent conversation (system prompt + tool schemas
                + a few tool results) blows past 2048 within a couple of steps, and the
                symptom is a model that suddenly "forgets" the task or its tools. So the
                budget is explicit and generous instead of left to a default that is
                wrong for agents.
            seed: the random seed for sampling. Temperature 0 makes the model pick its
                most likely token each time, but that is NOT full determinism — Ollama
                still draws from its own random state in a few places, and two runs of
                the same prompt can diverge. Pinning the seed as well removes the
                randomness we can control, which is what lets one benchmark run be
                re-run and compared with the last.
            temperature: sampling temperature; 0 for the reproducible benchmark.
            max_retries: how many times to re-send a request after its FIRST attempt
                fails — the same meaning the Anthropic SDK gives the same name, so
                a value means the same thing on both providers. 0 makes exactly
                one attempt; the default 3 allows up to four.
            backoff: seconds to wait before the first retry; doubles each time.
            num_predict: the per-reply output cap, in tokens. A model that leaks its
                chain of thought or loops can otherwise generate until the context is
                full; 2048 is far above any real tool call or answer.
            presence_penalty: sent explicitly because the library tag for qwen3.5:4b
                BAKES IN presence_penalty 1.5 in its Modelfile, and that setting made the
                model unable to re-emit text it had just read — exactly what rewriting a
                file with write_file is. With 1.5 it dropped the untouched functions and
                went 7/9 on the coding spike; with 0 it went 9/9. Request options
                override the Modelfile, so cornac sets 0 on every request.
            think: whether to ask the daemon for "thinking" (chain of thought). None
                means automatic: off for any Qwen 3 family tag (qwen3, qwen3.5, the
                Unsloth GGUFs), and the key is not sent at all for other models, whose
                templates may reject it. Thinking is off for the benchmark so that runs
                are comparable and the thinking text cannot leak into answers.
            extra_options: any further Ollama options, merged LAST so they can override
                the ones above (e.g. {"top_k": 0}). request_options() returns the merged
                dict — the benchmark records it on every run.
        """
        self.model = model
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.context_window = num_ctx  # what the loop sizes its clearing against (Week 4c)
        self.seed = seed
        self.temperature = temperature
        self.max_retries = max_retries
        self.backoff = backoff
        self.num_predict = num_predict
        self.presence_penalty = presence_penalty
        self.think = think
        self.extra_options = dict(extra_options or {})
        self._client = httpx.Client(timeout=timeout)
        # Kept as an attribute so tests can swap it for a no-op and not actually wait.
        self._sleep = time.sleep
        # How many tool-call ids this provider has made up so far (see _from_ollama).
        self._synthesized_ids = 0

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def request_options(self) -> dict:
        """The exact "options" dict sent with every request — see __init__ for each one.

        extra_options is merged last so a caller can override any default. The benchmark
        logs this dict per run: a result nobody can reproduce is not a result.
        """
        options = {
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "seed": self.seed,
            "num_predict": self.num_predict,
            "presence_penalty": self.presence_penalty,
        }
        options.update(self.extra_options)
        return options

    def think_flag(self) -> bool | None:
        """What to send as the top-level "think" field, or None to leave it out.

        Automatic mode turns thinking OFF for the Qwen 3 family (tag contains "qwen3":
        qwen3, qwen3.5, and the hf.co/unsloth GGUFs of them). Other models don't get
        the key at all — a template that has no thinking mode may reject it.
        """
        if self.think is not None:
            return self.think
        return False if "qwen3" in self.model.lower() else None

    def complete(self, messages: list[Message], tools: list[dict]) -> Message:
        payload: dict = {
            "model": self.model,
            "messages": self._to_ollama(messages),
            "stream": False,
            # Per-request generation settings — see __init__ for why each one matters.
            "options": self.request_options(),
        }
        think = self.think_flag()
        if think is not None:
            payload["think"] = think
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

        data = self._post_with_retries(f"{self.host}/api/chat", payload)
        return self._from_ollama(data)

    # --- HTTP with retries ---------------------------------------------------

    def _post_with_retries(self, url: str, payload: dict) -> dict:
        """POST `payload` to `url` and return the parsed JSON, retrying transient failures.

        A local Ollama is one process, not a cloud API with a fleet behind it. During a
        long benchmark it can be caught mid model-load or mid-swap, and the request
        fails with a dropped connection (httpx.TransportError: refused, reset, timed
        out) or a 5xx. Those clear within a second or so, so we wait and try again.

        A 4xx is different in kind: the request itself is wrong (unknown model,
        malformed payload) and resending the same bytes cannot fix it, so we raise at
        once rather than waste every retry on it.

        Waits are backoff, 2*backoff, 4*backoff, ... — doubling so a daemon that is
        genuinely busy gets progressively more breathing room instead of being hammered.
        """
        # "Retries" are the attempts AFTER the first one, as in the Anthropic SDK: a
        # value of 0 still makes the one request, it just never repeats it.
        attempts = 1 + max(0, self.max_retries)
        for attempt in range(attempts):
            try:
                resp = self._client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code < 500:
                    # A client error: retrying cannot help, so raise at once — but
                    # with the daemon's reason attached, not just the status line.
                    raise self._with_daemon_reason(e) from e
                error: Exception = self._with_daemon_reason(e)
            except httpx.TransportError as e:
                error = e

            # Don't sleep after the final attempt — there is nothing left to wait for.
            if attempt < attempts - 1:
                self._sleep(self.backoff * 2**attempt)

        # Every attempt failed: surface the last error rather than a vague one.
        raise error

    @staticmethod
    def _with_daemon_reason(e: httpx.HTTPStatusError) -> httpx.HTTPStatusError:
        """Rebuild an HTTP error so its message carries Ollama's own explanation.

        httpx's message is only the status line — "Server error '500 Internal Server
        Error'". Ollama, though, puts the real reason in the response body:
        {"error": "cudaMalloc failed: out of memory"} when the model won't fit on the
        GPU, {"error": "model 'x' not found"} for a typo in the tag, and so on. Dropping
        that text turns a one-line diagnosis into an afternoon of probing (we learned
        this the hard way). So the raised error says both: the status AND the daemon's
        words. The original request and response are kept on it, so callers can still
        inspect .response.status_code exactly as before.
        """
        resp = e.response
        reason = None
        try:
            body = resp.json()
            if isinstance(body, dict):
                reason = body.get("error")
        except ValueError:  # the body was not JSON (a proxy's HTML page, say)
            pass
        if not reason:
            reason = resp.text.strip()[:300] or "(empty response body)"
        return httpx.HTTPStatusError(
            f"Ollama returned {resp.status_code}: {reason}",
            request=e.request,
            response=resp,
        )

    # --- neutral -> Ollama ---------------------------------------------------

    def _to_ollama(self, messages: list[Message]) -> list[dict]:
        out: list[dict] = []
        for msg in messages:
            if msg.role in ("system", "user"):
                out.append({"role": msg.role, "content": msg.text})

            elif msg.role == "assistant":
                m: dict = {"role": "assistant", "content": msg.text or ""}
                if msg.tool_calls:
                    m["tool_calls"] = [
                        {"function": {"name": c.name, "arguments": c.arguments}}
                        for c in msg.tool_calls
                    ]
                out.append(m)

            elif msg.role == "tool":
                # Ollama matches tool results to calls by order/name, not by id.
                out.append({"role": "tool", "content": msg.text, "tool_name": msg.name})

        return out

    # --- Ollama -> neutral ---------------------------------------------------

    def _next_call_id(self, name: str) -> str:
        """A tool-call id unique for the life of this provider: call_<n>_<name>."""
        call_id = f"call_{self._synthesized_ids}_{name or 'tool'}"
        self._synthesized_ids += 1
        return call_id

    def _from_ollama(self, data: dict) -> Message:
        message = data.get("message", {})
        text = message.get("content", "") or ""

        tool_calls: list[ToolCall] = []
        for call in message.get("tool_calls", []) or []:
            fn = call.get("function", {})
            # Ollama returns arguments already parsed into a dict (unlike OpenAI's
            # JSON-string). It usually omits an id, so we synthesize one — numbered
            # across the whole conversation, not within the response. The first
            # version restarted at 0 on every reply, so every turn's first call was
            # `call_0_<name>`, and anything that matched results to calls by id (the
            # digest, a transcript reader) saw the last turn's outcome on every
            # earlier call with the same name.
            call_id = call.get("id") or self._next_call_id(fn.get("name", "tool"))
            tool_calls.append(
                ToolCall(id=call_id, name=fn.get("name", ""), arguments=dict(fn.get("arguments", {})))
            )

        # Token counts sit at the top level of the response, next to "message":
        # prompt_eval_count is what the model read, eval_count what it wrote. Either can
        # be missing (Ollama omits prompt_eval_count when the whole prompt was already
        # in its cache) or present-but-null, so both mean 0 rather than a crash. The
        # `or 0` matters: the agent adds usage up across the run, and int + None would
        # blow up the whole run() over a bookkeeping detail.
        usage = Usage(
            input_tokens=data.get("prompt_eval_count") or 0,
            output_tokens=data.get("eval_count") or 0,
        )

        return Message(
            role="assistant",
            text=text,
            tool_calls=tool_calls,
            usage=usage,
            # Ollama's own word for why it stopped: "stop" (finished naturally) or
            # "length" (ran into num_predict / the context limit). Kept verbatim.
            stop_reason=data.get("done_reason"),
        )
