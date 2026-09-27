"""Tests for the spike runner's own logic (benchmark/spike/run_spike.py).

The runner is a script, not part of the package, and until now nothing tested it
beyond "it compiles" and its round-three plumbing (test_workspace_hint.py). Three
claims its README makes are pinned here:

  - SpikeProvider layers its options in a fixed order — the provider's frozen
    defaults, then the runner's pinned temperature/num_ctx/seed/num_predict, then
    --options last — and the dict it records is the dict it sends.
  - When the daemon rejects the `think` flag for a tag (a 400 that names it), the
    request is re-sent once without the flag and the tag remembered, so one 400 does
    not become k failed runs.
  - The announced-then-stopped column uses the agent's own nudge trigger, so it and
    the nudges column count the same phrases.

No Ollama is involved: the HTTP layer is an httpx.MockTransport, and the `spike`
fixture (conftest.py) loads the runner as a module.
"""

from __future__ import annotations

import json

import httpx
import pytest

from cornac.core.agent import announces_next_step
from cornac.core.messages import Message
from cornac.tools.builtin import default_tools
from cornac.tools.workspace import Workspace
from test_nudge import ANNOUNCING, PLAIN


# --- option layering ---------------------------------------------------------------

def test_request_options_layer_defaults_then_runner_then_override(spike):
    provider = spike.make_provider(
        "qwen3.5:4b", seed=3, temperature=0.3, extra_options={"num_predict": 512}
    )
    options = provider.request_options()
    assert options["presence_penalty"] == 0.0     # the provider's frozen default
    assert options["seed"] == 3                   # the runner's pins
    assert options["temperature"] == 0.3
    assert options["num_ctx"] == spike.NUM_CTX
    assert options["num_predict"] == 512          # --options wins over NUM_PREDICT


def test_without_an_override_the_runner_cap_stands(spike):
    options = spike.make_provider("qwen3.5:4b", seed=1, temperature=0.0).request_options()
    assert options["num_predict"] == spike.NUM_PREDICT


def test_the_recorded_options_are_the_sent_options(spike, think_rejected):
    # run_once records provider.request_options(); the request must carry the same dict.
    provider = spike.make_provider("qwen3.5:4b", seed=7, temperature=0.3)
    seen = _fake_daemon(provider, [(200, _chat_reply())])
    provider.complete([Message.user("hi")], [])
    assert seen[0]["options"] == provider.request_options()


def test_make_provider_makes_one_attempt(spike):
    assert spike.make_provider("m", seed=1, temperature=0.0).max_retries == 0


# --- the think re-send ---------------------------------------------------------------

@pytest.fixture
def think_rejected(spike):
    """THINK_REJECTED empty before and after: it is module state that would leak."""
    spike.THINK_REJECTED.clear()
    yield spike.THINK_REJECTED
    spike.THINK_REJECTED.clear()


def _chat_reply(text: str = "ready") -> dict:
    return {
        "message": {"role": "assistant", "content": text},
        "done_reason": "stop",
        "prompt_eval_count": 1,
        "eval_count": 1,
    }


def _fake_daemon(provider, responses: list[tuple[int, dict]]) -> list[dict]:
    """Point `provider` at a scripted daemon; returns the request payloads it receives."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        status, body = responses.pop(0)
        return httpx.Response(status, json=body)

    provider._client = httpx.Client(transport=httpx.MockTransport(handler))
    return seen


def test_a_400_naming_think_is_resent_once_without_the_flag(spike, think_rejected):
    provider = spike.make_provider("qwen3.5:4b", seed=1, temperature=0.0)
    seen = _fake_daemon(provider, [
        (400, {"error": "this model does not support thinking"}),
        (200, _chat_reply()),
    ])
    reply = provider.complete([Message.user("hi")], [])

    assert reply.text == "ready"
    assert seen[0]["think"] is False              # the first request carried the flag
    assert "think" not in seen[1]                 # the re-send dropped it
    assert think_rejected == {"qwen3.5:4b"}


def test_a_rejected_tag_sends_no_think_flag_afterwards(spike, think_rejected):
    think_rejected.add("qwen3.5:4b")
    provider = spike.make_provider("qwen3.5:4b", seed=1, temperature=0.0)
    seen = _fake_daemon(provider, [(200, _chat_reply())])
    provider.complete([Message.user("hi")], [])
    assert len(seen) == 1 and "think" not in seen[0]


def test_a_400_for_another_reason_is_raised_not_resent(spike, think_rejected):
    provider = spike.make_provider("qwen3.5:4b", seed=1, temperature=0.0)
    seen = _fake_daemon(provider, [(400, {"error": "model 'qwen3.5:4b' not found"})])
    with pytest.raises(httpx.HTTPStatusError, match="not found"):
        provider.complete([Message.user("hi")], [])
    assert len(seen) == 1
    assert not think_rejected


# --- announced-then-stopped agrees with the nudge ------------------------------------

@pytest.mark.parametrize("text", ANNOUNCING + PLAIN, ids=repr)
def test_announced_then_stopped_agrees_with_the_agents_detector(spike, text):
    assert spike.announced_then_stopped("done", False, text) == announces_next_step(text)


def test_announced_then_stopped_needs_a_failed_done_run(spike):
    text = "Let's fix it and run the tests again."
    assert spike.announced_then_stopped("done", False, text)
    assert not spike.announced_then_stopped("done", True, text)         # a pass is not a stall
    assert not spike.announced_then_stopped("max_steps", False, text)   # the last message called a tool
    assert not spike.announced_then_stopped("error", False, text)       # there was no final message
    assert not spike.announced_then_stopped("done", False, None)


def test_the_runner_keeps_no_phrase_list_of_its_own(spike):
    # One detector, in the agent. A second list here drifted from it once already.
    assert not hasattr(spike, "ANNOUNCE_RE")


# --- the advertised tool set (Week 4b-B) ---------------------------------------------

def test_the_spike_advertises_the_ten_tools_the_model_was_frozen_with(spike, tmp_path):
    # spawn_agent joined default_tools after the freeze commit (c381aed). The spike
    # leaves it out on purpose, so the frozen model keeps seeing the schema it was
    # frozen against and no run wastes a step on a spawn the policy would deny.
    names = [t.name for t in spike.spike_tools(Workspace(tmp_path))]

    assert names == ["read_file", "write_file", "edit_file", "list_dir", "grep",
                     "run_bash", "run_python", "web_search", "web_fetch", "get_current_time"]
    assert "spawn_agent" in [t.name for t in default_tools(Workspace(tmp_path))]
    # Nothing the model can see is left for the policy's default deny to swallow
    # silently: every advertised tool has an explicit rule.
    assert set(names) <= set(spike.POLICY_RULES["tools"])


# --- the Week 4c loop behaviours are pinned off, not inherited -------------------------

def test_the_runner_pins_the_week_4c_loop_behaviours_off_and_records_them(spike, tmp_path):
    # OllamaProvider tells the loop its num_ctx, so an Agent built with defaults would
    # clear context at 75% of 8192, stop "stuck" after three identical calls and make a
    # wrap-up call on every unfinished run — none of which round three was measured
    # with. build_agent switches all three off explicitly; the provider still knows
    # its window, which is what proves the pin rather than an absent value.
    provider = spike.make_provider("qwen3.5:4b", seed=1, temperature=0.0)
    agent = spike.build_agent(provider, Workspace(tmp_path))

    assert spike.HARNESS_SETTINGS == {"max_repeats": 0, "wrap_up": False, "context_window": 0}
    assert (agent.max_repeats, agent.wrap_up, agent.context_window) == (0, False, 0)
    assert provider.context_window == spike.NUM_CTX          # inherited it would have been 8192
    assert agent.max_steps == spike.MAX_STEPS
    assert agent.max_nudges == 1                              # the rest stays the Agent's default
