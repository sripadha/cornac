#!/usr/bin/env python
"""Model spike: pick the benchmark's local model on evidence, then freeze it.

The plan (docs/plan.md, "spike-then-freeze") fixes the benchmark on ONE local model
for the whole capability ladder. Which one is an empirical question — Qwen 2.5 7B has
been the development model, but Qwen 3 (8B and 4B) exist and claim better tool
calling — and the wrong way to settle it is by reputation. This script settles it with
data: the same small tasks on every candidate, k runs each, graded by each task's
verify.py. Whatever calls tools most reliably wins, and the choice is then frozen:
the point of a spike is to decide once and stop churning.

Run it from the repo root with Ollama up:

    python benchmark/spike/run_spike.py                       # every task, k=3
    python benchmark/spike/run_spike.py --models qwen3:4b --k 1
    # round two (coding only, sampling on, its own results directory):
    python benchmark/spike/run_spike.py --tasks swe,swe2,swe3 --temperature 0.3 \
        --models qwen2.5:7b-instruct-q4_K_M,qwen3:8b,qwen3:4b-instruct-2507-q4_K_M \
        --out benchmark/spike/results_round2

It writes <out>/runs.jsonl (one JSON line per run — the raw record), <out>/models.json
(per-model facts, the tool-call probe included), <out>/transcripts/ (every run's full
conversation, plus each model's probe) and <out>/results.md (the table), and prints
the table.

Design notes, so the numbers can be trusted:

  - Outer loop over models. Only one model fits the 6 GB GPU at a time and Ollama
    loads a model on its first request, so running all of one model's tasks back to
    back loads each model once instead of swapping on every run.
  - Every run gets a FRESH copy of the task's fixture in a temp dir. The fixture is
    never run in place, so a run can never see what an earlier run wrote.
  - Same system prompt, tools, policy, max_steps and context window for every model.
    The only variable is the model tag — plus think=False for Qwen 3, which is the
    documented way to run it as a plain instruct model (see SpikeProvider).
  - A provider error (an Ollama 500, a timeout) is recorded as a failed run, not
    raised: a spike that dies on run 7 of 27 wastes the first six.
  - Runs APPEND to runs.jsonl and results.md is rebuilt from the whole file, so a
    model can be added later ("--models qwen3:8b" once it is pulled) and the table
    stays complete. Delete the out directory for a clean slate.
  - Every model gets the same output cap per step (NUM_PREDICT) and no HTTP retries,
    so a runaway or timed-out step costs seconds, not tens of minutes, and is
    recorded as what it was rather than folded into the averages. See SpikeProvider
    and make_provider.
  - A run in which the conversation outgrew the context window is flagged
    (context_overflow). Ollama truncates the prompt silently in that case, and a
    failure that follows is a budget problem, not evidence about tool calling; it
    must be labelled, not booked as plain model behavior.

Round two (coding focus) added, on the lessons of round one:

  - --temperature (default 0.0, unchanged) and --seed-base (default 42, unchanged).
    At temperature 0 the three seeds of round one produced byte-identical runs, so
    k=3 measured nothing; round two samples at 0.3 with seeds 42/43/44, and the
    temperature is recorded on every runs.jsonl line so a mixed file stays honest.
  - A tool-call PROBE before each model's runs: one tiny task with only
    get_current_time registered. A model that cannot emit a tool call through Ollama
    at all is disqualified for this harness, and the reason must be visible: the
    verdict is on every run line (tool_capable), in models.json with the raw text of
    the probe reply (where leaked <think> text shows), and as a warning in the table.
    Its tasks are still run, so the rows exist for the record.
  - announced_then_stopped, a HEURISTIC flag: the run ended with a final message that
    had no tool calls, the task did not pass, and the text says something like "let
    me fix it and re-run" — the failure mode that accounted for all four 7-8B swe
    failures in round one. It is a phrase match, nothing more. Since round three it
    is the agent's own detector (cornac.core.agent.announces_next_step, which reads
    the last line of the reply), so this column and the `nudges` column count the
    same phrases; the runner used to keep a list of its own, with a bare "next"
    that also flagged honest summaries.
  - Model tags may contain "/" and ":" (hf.co/unsloth/...:Q8_0). The raw tag is what
    goes to Ollama and into the table; only file names are sanitized (safe_name).
  - think=False goes to any tag whose lowercase name contains "qwen3" (qwen3, qwen3.5
    and the unsloth GGUFs), and if the daemon rejects the flag for a tag, the flag is
    dropped for that tag and the request re-sent once (see SpikeProvider).

Round two-b (a sampling check before freezing the model) added:

  - --options, a JSON object merged into every request's Ollama "options" AFTER the
    runner's own temperature/num_ctx/seed/num_predict, so it can override them. It
    overrides the tag's Modelfile as well: a per-request option beats a PARAMETER
    line. The merged dict goes on every runs.jsonl line ("options") and the flag's
    own value as "options_override", and the tables keep one row per (model,
    override), so a tag run with and without an override is never averaged into
    one row. The how-to-reproduce block prints one command per override.
  - Per model, the PARAMETER lines of `ollama show <tag> --modelfile` are recorded in
    models.json ("modelfile_parameters") and shown in the facts table. The library
    tag qwen3.5:4b bakes in presence_penalty 1.5, top_k 20, top_p 0.95 and
    temperature 1; the HF GGUF imports bake in nothing. A row cannot be read
    without knowing this: round two's qwen3.5:4b failures were write_file payloads
    that stopped after the first function of a file the model had just read, which
    is what a presence penalty of 1.5 does to text already in the context.

Round three (the harness under test, on the frozen model) changes the runner, not
the tasks. Rounds two and two-b showed that most coding failures were the HARNESS's:
whole-file write_file rewrites that dropped untouched functions or broke a closing
triple-quote; a broken file discovered only by the next pytest and then re-sent
unchanged; runs that ended by announcing the next step and never taking it; a system
prompt that never said where the workspace was (one model ran `cd /workspace`); a
function "fixed" inside run_python and believed to be on disk. Week 4b closes those
gaps in cornac (edit_file, a syntax check after write_file, a nudge for a model that
announces and stops, an honest write_file summary, Workspace.describe(), tool
descriptions that say what run_bash/run_python do NOT do), and round three measures
them: same tasks, same k, same frozen model. In the runner:

  - The default --models is the frozen tag, qwen3.5:4b. The provider's own defaults
    (presence_penalty 0, num_predict 2048, think off for the Qwen 3 family) are the
    frozen ones now, so no --options override is needed; the `options` field of every
    run line is the dict the provider actually sent (SpikeProvider.request_options).
  - The spike policy allows edit_file next to write_file. The tool itself comes from
    default_tools, like every other built-in.
  - The per-run system prompt is SYSTEM_PROMPT plus Workspace.describe() for that
    run's temp directory (system_prompt_for), so the model is told its absolute root.
  - The Agent is built with its defaults for everything the spike does not need to
    pin — max_nudges included — so a run measures the harness as shipped. The number
    of nudges a run needed is on every run line (`nudges`; 0 for older lines that
    predate the field) and in a "nudges" column of the table.

Week 4c added three behaviours to the loop that would otherwise arrive here silently:
the repeat hard stop (a run ends "stuck" after three identical calls with identical
results), the wrap-up call (one extra model call on an unfinished run) and context
clearing (OllamaProvider now tells the loop its num_ctx, so clearing would switch
itself on at 75% of 8192). Round three — the 12/27 -> 26/27 result — was measured
without them, and replaying its transcripts under the new rules changes the one
qwen3:8b failure from max_steps to "stuck", adds a wrap-up reply as the last
assistant message of every unfinished run (which is what `last_stop_reason` and the
token totals read), and puts one round-two run within 2% of the clearing trigger. So
the runner PINS all three off (HARNESS_SETTINGS, passed by build_agent) and records
the three values on every run line. A rung that turns them on is a new, labelled
measurement, not a different harness under the same label.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import httpx

SPIKE_DIR = Path(__file__).resolve().parent
TASKS_DIR = SPIKE_DIR / "tasks"
REPO_ROOT = SPIKE_DIR.parent.parent

# `python benchmark/spike/run_spike.py` puts benchmark/spike on sys.path, not the repo
# root, and cornac is not pip-installed in the dev venv. Adding the root makes the
# imports below work from any cwd without an editable install.
sys.path.insert(0, str(REPO_ROOT))

from cornac import Agent, Message, Policy, Tool, ToolRegistry  # noqa: E402
from cornac.core.agent import announces_next_step  # noqa: E402
from cornac.providers.ollama import OllamaProvider  # noqa: E402
from cornac.tools.builtin import default_tools, get_current_time  # noqa: E402
from cornac.tools.workspace import Workspace  # noqa: E402

# The frozen benchmark model (round two-b); rounds one and two passed --models by hand.
DEFAULT_MODELS = "qwen3.5:4b"
DEFAULT_OUT = SPIKE_DIR / "results"
# Table order: the benchmark's three domains, with the round-two coding tasks
# right after the original swe task.
TASK_ORDER = ["data", "swe", "swe2", "swe3", "research"]
# The tools that can actually do arithmetic; the "data via tool" column counts data
# runs that invoked at least one of them.
COMPUTE_TOOLS = {"run_python", "run_bash"}

SYSTEM_PROMPT = (
    "You are a careful assistant working in a sandboxed workspace. Use the tools to "
    "look at the files before answering. Do not guess numbers - compute them. When "
    "done, state the final answer plainly."
)
MAX_STEPS = 12
NUM_CTX = 8192
# The Week 4c loop behaviours, pinned OFF for the frozen-model rungs: see the module
# docstring. max_repeats 0 = no hard stop (the Week 4b advisory note still fires),
# wrap_up False = no post-mortem model call, context_window 0 = clearing off even
# though the provider knows its window. Recorded on every runs.jsonl line.
HARNESS_SETTINGS: dict = {"max_repeats": 0, "wrap_up": False, "context_window": 0}
# Output tokens per step, for every model. A tool call is a few dozen tokens and a
# final answer a few hundred (Qwen 2.5 peaked at 211 in the smoke test); 2048 is far
# above either and exists only to stop a step that rambles. See SpikeProvider.
NUM_PREDICT = 2048
DEFAULT_SEED_BASE = 42
DEFAULT_TEMPERATURE = 0.0
REQUEST_TIMEOUT = 300  # seconds per model call; a 7B model on a laptop can need a while


def build_agent(provider: OllamaProvider, workspace: Workspace) -> Agent:
    """The Agent every spike run uses — the one place its harness settings are chosen.

    Only what the spike must pin is passed. Everything else — the nudge limit
    (max_nudges), hooks, the approver — is the Agent's own default, so a run
    measures the harness as shipped. The exception is HARNESS_SETTINGS: the three
    Week 4c behaviours are switched off explicitly, because the frozen rung was
    measured without them and a default that changes under the same label is not
    a measurement (module docstring).
    """
    return Agent(
        provider=provider,
        registry=ToolRegistry(spike_tools(workspace)),  # sub-agents (Week 4b-B)
        system_prompt=system_prompt_for(workspace),
        max_steps=MAX_STEPS,
        policy=Policy(POLICY_RULES),
        **HARNESS_SETTINGS,
    )


def system_prompt_for(workspace: Workspace) -> str:
    """SYSTEM_PROMPT plus Workspace.describe(): the fixed text, then where THIS run lives.

    Every run gets a fresh temp directory, so the paragraph naming the absolute root
    has to be built per run. Round two had a model run `cd /workspace` (no such
    directory) and round one had one run pytest on a guessed path; nothing had told
    them where they were. The same paragraph goes to every model, so it is part of
    the fixed conditions, not a per-model tweak.
    """
    return SYSTEM_PROMPT + "\n\n" + workspace.describe()


# The tool-call probe: the smallest possible agent task, with one tool registered.
# A model cannot know the wall-clock time from its weights, so the only way to answer
# is to call the tool — and whether ANY tool call comes out is all the probe records.
PROBE_PROMPT = "What is the current time in Tokyo? Use the tool."
PROBE_SYSTEM_PROMPT = "You are a helpful assistant with tools. Call a tool when it helps."
PROBE_MAX_STEPS = 3

# One policy for every model. The spike is headless — nobody is there to answer an
# ASK, and the agent turns an unanswered ASK into a DENY — so every tool a task can
# need is a plain allow. run_bash keeps the destructive-command blocklist: the model
# runs real shell commands in a real temp dir, and "it was only a benchmark" is no
# excuse for `rm -rf`. The two web tools are denied: every spike task is offline, and
# a run that reaches for the network is both a wrong move and a non-reproducible one.
POLICY_RULES: dict = {
    "default": "deny",
    "tools": {
        "read_file": "allow",
        "write_file": "allow",
        # Round three: the few-lines edit tool, next to the whole-file one. Round two's
        # rewrites dropped functions and broke docstrings because write_file was the
        # only way to change a file; whether edit_file gets used is what this measures.
        "edit_file": "allow",
        "list_dir": "allow",
        "grep": "allow",
        "run_python": "allow",
        "get_current_time": "allow",
        "run_bash": {
            "deny": ["rm -rf", "sudo", "mkfs", "dd if=", ":(){"],
            "default": "allow",
        },
        "web_search": "deny",
        "web_fetch": "deny",
    },
}

# --- sub-agents (Week 4b-B) ---------------------------------------------------
# Built-ins the spike does NOT advertise. spawn_agent joined default_tools after the
# model was frozen (commit c381aed measured qwen3.5:4b against the ten tools before
# it), and POLICY_RULES would deny it anyway (default: deny, spawn_agent unlisted):
# leaving it in would change the tool schema the frozen rounds saw and turn every
# attempt to delegate into a wasted "Permission denied" step. Giving the spike
# sub-agents is a deliberate change for a later round — drop the name here, allow
# it in POLICY_RULES, and record the change in docs/observations.md.
SPIKE_EXCLUDED_TOOLS = frozenset({"spawn_agent"})


def spike_tools(workspace: Workspace) -> list[Tool]:
    """The tools a spike run advertises: default_tools minus SPIKE_EXCLUDED_TOOLS."""
    return [t for t in default_tools(workspace) if t.name not in SPIKE_EXCLUDED_TOOLS]
# --- end sub-agents ------------------------------------------------------------


# --- the provider tweak --------------------------------------------------------


# Tags for which the daemon rejected the `think` flag. Filled at runtime; a tag lands
# here once and its later requests go out without the flag (see SpikeProvider).
THINK_REJECTED: set[str] = set()


def runner_options(seed: int, temperature: float, extra: dict) -> dict:
    """The spike's own generation settings, in merge order, with --options last.

    These go OVER the provider's defaults (SpikeProvider.request_options), so the
    spike pins the same temperature, context window, seed and per-step output cap
    for every model whatever the provider's defaults become. `extra` comes last, so
    `--options '{"num_predict": 512}'` wins over NUM_PREDICT — that is the point of
    the flag.
    """
    return {
        "temperature": temperature,
        "num_ctx": NUM_CTX,
        "seed": seed,
        "num_predict": NUM_PREDICT,
        **extra,
    }


def parse_options(text: str | None) -> dict:
    """The --options flag as a dict: a JSON object, or {} when the flag was not given."""
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--options is not valid JSON ({exc}): {text!r}") from exc
    if not isinstance(value, dict):
        raise SystemExit(
            f"--options must be a JSON object such as '{{\"presence_penalty\": 0}}', got {text!r}"
        )
    return value


class SpikeProvider(OllamaProvider):
    """OllamaProvider with two spike-specific behaviors.

    1. One `options` dict that is both sent and recorded. request_options() is the
       provider's frozen defaults (presence_penalty 0 among them), then the runner's
       own settings (runner_options), then the --options overrides last. Ollama
       applies a request's options over the tag's Modelfile PARAMETER lines, so
       --options is the one way to switch off a setting a tag bakes in (round two-b
       did that for qwen3.5:4b's presence_penalty 1.5) without building a new tag.
       The base provider calls this method to build every request, and run_once
       calls it to fill the `options` field of the run line, so the record and the
       request agree by construction.

       Why the runner re-states a cap the provider already has: think=False is not
       the whole story. With it, Ollama 0.30.9 returns no `thinking` field for
       qwen3:4b, but the model still writes its chain of thought as ordinary
       `content` ("Okay, let's see..."). Measured: 1,918 output tokens before its
       first tool call in this runner, 5,689 in a direct probe, 56 s for a two-step
       data run. Uncapped, one rambling step eats minutes, and at ~2k tokens a step
       a 7-12 step run blows through NUM_CTX, after which Ollama silently truncates
       the prompt. The cap turns a runaway step into done_reason "length" on that
       message: recorded (last_stop_reason), visible, and cheap. It is pinned here
       for EVERY model so the comparison stays fair even if the provider's default
       moves; Qwen 2.5 never used more than 211 output tokens in a step during the
       smoke test, so it never notices.

    2. A fallback for the `think` flag. The base provider sends think=False for any
       tag whose lowercase name contains "qwen3" (qwen3, qwen3.5 and the unsloth
       GGUFs), which runs the model as a plain instruct model — the fair comparison
       and the mode the plan specifies. A tag whose template lacks the thinking
       capability may reject the flag with a 400; the instruct-2507 tags and the
       GGUF imports are exactly the ones that may. When that happens the tag is
       remembered (THINK_REJECTED), the flag dropped, and the request re-sent once —
       one 400 must not become k failed runs.

    Overriding request_options and the HTTP step keeps every other part of the
    payload identical to what the shipped provider sends.
    """

    def request_options(self) -> dict:
        return {
            **super().request_options(),
            **runner_options(self.seed, self.temperature, self.extra_options),
        }

    def _post_with_retries(self, url: str, payload: dict) -> dict:
        if self.model in THINK_REJECTED:
            payload.pop("think", None)
        try:
            return super()._post_with_retries(url, payload)
        except httpx.HTTPStatusError as exc:
            rejected = exc.response.status_code == 400 and "think" in str(exc).lower()
            if "think" in payload and rejected:
                THINK_REJECTED.add(self.model)
                print(
                    f"[{self.model}] daemon rejected think={payload['think']} ({exc}); "
                    "re-sending without the flag for this tag",
                    file=sys.stderr,
                    flush=True,
                )
                payload.pop("think", None)
                return super()._post_with_retries(url, payload)
            raise


def make_provider(
    model: str, seed: int, temperature: float, extra_options: dict | None = None
) -> SpikeProvider:
    """The one place the generation settings are chosen, so every call agrees."""
    return SpikeProvider(
        model=model,
        num_ctx=NUM_CTX,
        seed=seed,
        temperature=temperature,
        timeout=REQUEST_TIMEOUT,
        extra_options=extra_options,
        # No retries. OllamaProvider's default (3, with backoff) suits a long
        # benchmark and is wrong for a timed spike: httpx.ReadTimeout is a
        # TransportError, so a step that hits REQUEST_TIMEOUT would be re-sent up to
        # three more times — twenty minutes for one step — and if a retry then
        # succeeded, the lost minutes would fold silently into `duration`, `avg secs`
        # and tokens/sec with no error on record. With one attempt, a timeout is
        # recorded once, as `error`, and shows in the errors column. The warm-up
        # already absorbs model-load hiccups, and on a single-model daemon a 5xx in
        # the middle of a spike is not transient.
        max_retries=0,
    )


def safe_name(model: str) -> str:
    """A model tag as a file-name stem: every character outside [A-Za-z0-9._-] -> "_".

    `qwen2.5:7b-instruct-q4_K_M` -> `qwen2.5_7b-instruct-q4_K_M` (the same stem
    round one used, so old and new transcripts sit side by side);
    `hf.co/unsloth/Qwen3-4B-GGUF:Q8_0` -> `hf.co_unsloth_Qwen3-4B-GGUF_Q8_0`. Only
    file names go through this; the raw tag is what Ollama and the table see.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


# --- tasks -------------------------------------------------------------------------


@dataclass
class Task:
    name: str
    prompt: str
    fixture: Path
    verify: Callable[[str, str], tuple[bool, str]]


def load_task(task_dir: Path) -> Task:
    """Read one task directory: the prompt from task.md, verify() from verify.py."""
    text = (task_dir / "task.md").read_text(encoding="utf-8")
    # The prompt is the body of the "## Prompt" section, up to the next "## " heading.
    # task.md is the single source of truth for what the model was asked, so a reader
    # who opens it sees exactly the prompt — there is no second copy in this script.
    match = re.search(r"^## Prompt\s*\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    if not match:
        raise ValueError(f"{task_dir / 'task.md'} has no '## Prompt' section")
    prompt = " ".join(match.group(1).split())  # the paragraph may be hard-wrapped

    spec = importlib.util.spec_from_file_location(
        f"spike_verify_{task_dir.name}", task_dir / "verify.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    return Task(
        name=task_dir.name,
        prompt=prompt,
        fixture=task_dir / "fixture",
        verify=module.verify,
    )


def discover_tasks(names: list[str] | None) -> list[Task]:
    """The requested tasks (or every task under tasks/), in table order."""
    available = {p.name: p for p in TASKS_DIR.iterdir() if (p / "task.md").is_file()}
    if names is None:
        names = sorted(available, key=_task_sort_key)
    missing = [n for n in names if n not in available]
    if missing:
        raise SystemExit(f"unknown task(s) {missing}; available: {sorted(available)}")
    return [load_task(available[n]) for n in names]


def _task_sort_key(name: str) -> tuple[int, str]:
    """Known tasks first, in TASK_ORDER; anything else alphabetically after."""
    return (TASK_ORDER.index(name) if name in TASK_ORDER else len(TASK_ORDER), name)


# --- Ollama facts ------------------------------------------------------------------


def ollama_column(subcommand: str, model: str, column: str) -> str | None:
    """Read one cell of `ollama ps` / `ollama list` for `model`, or None if not found.

    Both commands print an aligned table whose columns are separated by two or more
    spaces. A cell like "17%/83% CPU/GPU" contains single spaces, so splitting on
    any whitespace would break it; splitting on runs of two or more keeps cells whole.
    The header row names the columns, so the wanted one is looked up there and the
    same slot is taken from the model's row.
    """
    if shutil.which("ollama") is None:
        return None
    try:
        proc = subprocess.run(
            ["ollama", subcommand], capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if proc.returncode != 0 or not lines:
        return None
    header = re.split(r"\s{2,}", lines[0].strip())
    if column not in header:
        return None
    index = header.index(column)
    for line in lines[1:]:
        cells = re.split(r"\s{2,}", line.strip())
        if cells[0] == model and index < len(cells):
            return cells[index]
    return None


def gpu_percent(placement: str | None) -> int | None:
    """The GPU share of an `ollama ps` PROCESSOR cell.

    "100% GPU" -> 100, "17%/83% CPU/GPU" -> 83, "100% CPU" -> 0, unknown -> None.
    """
    if not placement:
        return None
    split = re.match(r"(\d+)%/(\d+)%\s+CPU/GPU", placement)
    if split:
        return int(split.group(2))
    whole = re.match(r"(\d+)%\s+(GPU|CPU)", placement)
    if whole:
        return int(whole.group(1)) if whole.group(2) == "GPU" else 0
    return None


def modelfile_parameters(model: str) -> list[str] | None:
    """The PARAMETER lines baked into `model`'s Modelfile, as "name value" strings.

    `ollama show <tag> --modelfile` prints the Modelfile the tag was built from. Its
    PARAMETER lines are sampling settings the tag carries on its own and applies to
    every request that does not set the same key — the library tag qwen3.5:4b ships
    `presence_penalty 1.5`, `top_k 20`, `top_p 0.95` and `temperature 1`, while an HF
    GGUF import ships none. Two tags of the same weights can therefore sample quite
    differently, and a row in the table cannot be read without knowing which settings
    were in force; --options is how one of them is overridden per request. [] means
    the Modelfile has no PARAMETER line; None means the command failed (the tag is
    not pulled, or ollama is not on PATH). Repeated keys (several `stop` lines) are
    kept as they are.
    """
    if shutil.which("ollama") is None:
        return None
    try:
        proc = subprocess.run(
            ["ollama", "show", model, "--modelfile"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    prefix = "PARAMETER "
    return [
        line[len(prefix):].strip() for line in proc.stdout.splitlines() if line.startswith(prefix)
    ]


def warm_up(
    model: str, seed: int, temperature: float, extra_options: dict | None = None
) -> str | None:
    """Load `model` with one tiny request, so the first real run does not pay for it.

    Every run's duration feeds the avg-seconds and tokens/sec columns. Without a
    warm-up, the first run of each model would carry 10-20 s of model loading that the
    other runs do not, and that noise would land on whichever task happens to go
    first. The same NUM_CTX as the runs is used because Ollama reloads a model when
    the context size changes; a warm-up at a different size would just be a second
    load. Returns the error text on failure — a model that is not pulled shows up
    here, once, before it fails k times in the table.
    """
    provider = make_provider(model, seed, temperature, extra_options)
    try:
        provider.complete([Message.user("Reply with the single word: ready")], [])
    except Exception as exc:  # noqa: BLE001 — reported, not fatal
        return f"{type(exc).__name__}: {exc}"
    return None


# --- the tool-call probe -----------------------------------------------------------


def probe_tool_calling(
    model: str,
    seed: int,
    temperature: float,
    transcript_dir: Path,
    extra_options: dict | None = None,
) -> dict:
    """Can `model` emit a tool call through Ollama at all? One tiny task, one tool.

    The probe registers only get_current_time and asks for the time in Tokyo. The
    verdict (tool_capable) is whether ANY assistant message in that short run carried
    a tool call: a tool call that is malformed, denied or errors still counts,
    because the question is whether tool calls come out of the model at all, not
    whether they were right. A model that fails the probe is disqualified for this
    harness — a tool-calling benchmark cannot measure a model that cannot call
    tools — but its tasks still run, so the rows exist and the table can say why
    they look the way they do.

    The raw text of the reply is kept too (first 200 characters): when a model's
    template ignores think=False, its chain of thought leaks into `content` (round
    one's qwen3:4b opened every reply with "Okay, let's see...") and this is where
    that becomes visible without opening a transcript. The first assistant message is
    taken when it has text (that is where prose-instead-of-a-call shows); otherwise
    the last one (a capable model's first message is often a bare tool call).
    """
    info: dict = {
        "tool_capable": False,
        "probe_reply": "",
        "probe_error": None,
        "probe_transcript": None,
    }
    agent = Agent(
        provider=make_provider(model, seed, temperature, extra_options),
        registry=ToolRegistry([get_current_time]),
        system_prompt=PROBE_SYSTEM_PROMPT,
        max_steps=PROBE_MAX_STEPS,
        policy=Policy(POLICY_RULES),
    )
    try:
        agent.run(PROBE_PROMPT)
    except Exception as exc:  # noqa: BLE001 — a probe that dies is data, not a crash
        info["probe_error"] = f"{type(exc).__name__}: {exc}"[:300]

    assistant = [m for m in agent.messages if m.role == "assistant"]
    info["tool_capable"] = any(m.tool_calls for m in assistant)
    if assistant:
        raw = assistant[0].text or assistant[-1].text or ""
        info["probe_reply"] = raw[:200]

    path = transcript_dir / f"{safe_name(model)}__probe.json"
    path.write_text(
        json.dumps(transcript(agent.messages), indent=2, default=str), encoding="utf-8"
    )
    info["probe_transcript"] = str(path.relative_to(transcript_dir.parent))
    return info


# --- one run -----------------------------------------------------------------------


def summarize_messages(messages: list[Message], num_ctx: int = NUM_CTX) -> dict:
    """Tool-use counts (and, as a fallback, steps/tokens) read off the conversation.

    A finished run's RunResult already carries steps and tokens; the fallbacks matter
    for a run that raised mid-way, where there is no RunResult but the messages up to
    the crash are still on the agent. The tool-use counts come only from here: the
    RunResult does not know which tools were called or how many calls errored.
    `num_ctx` is the window the run actually asked for (--options may override
    NUM_CTX), which is what the overflow check has to compare against.
    """
    assistant = [m for m in messages if m.role == "assistant"]
    results = [m for m in messages if m.role == "tool"]
    calls = [call.name for m in assistant for call in m.tool_calls]
    return {
        "steps": len(assistant),
        "input_tokens": sum(m.usage.input_tokens for m in assistant if m.usage),
        "output_tokens": sum(m.usage.output_tokens for m in assistant if m.usage),
        "n_tool_calls": len(calls),
        # Two kinds of bad tool call, kept apart because they fail differently. A
        # hard error is one the harness flagged (is_error=True): the policy denied
        # the call, or the tool raised — a KeyError from a misnamed argument, say. A
        # soft error is a tool that ran to completion and reported failure in its
        # text: "Error: not a file: sales.cvs" from read_file, "Error: not a
        # directory" from list_dir, "Error: command timed out" from run_bash. Both
        # mean the model asked for something that did not work, so the table's
        # tool-error rate counts both; runs.jsonl keeps them separate.
        "n_tool_errors": sum(1 for m in results if m.is_error),
        "n_soft_errors": sum(
            1 for m in results if not m.is_error and m.text.startswith("Error:")
        ),
        "n_distinct_tools": len(set(calls)),
        # Which tools were called at all, as a sorted set. On the data task this is
        # the evidence that run_python or run_bash did the arithmetic.
        "tools_used": sorted(set(calls)),
        "last_stop_reason": assistant[-1].stop_reason if assistant else None,
        # Ollama's prompt_eval_count is the FULL prompt of that call — the smoke test
        # showed it climbing 838 -> 952 -> 1131 across one run's steps — so prompt
        # plus output on any single call reaching NUM_CTX means the window was full
        # and Ollama truncated the prompt. It does so silently: no error, no field in
        # the response, only a warning in the daemon's own log. Whatever the model
        # did after that was done with part of its conversation missing, so a failure
        # that follows is a context-budget problem, not evidence about tool calling.
        # (Ollama omits prompt_eval_count when a prompt is entirely cached; that
        # cannot happen mid-run, where every step's prompt is longer than the last.)
        "context_overflow": any(
            m.usage.input_tokens + m.usage.output_tokens >= num_ctx
            for m in assistant
            if m.usage
        ),
    }


def transcript(messages: list[Message]) -> list[dict]:
    """The conversation as plain dicts, for the per-run transcript file."""
    out: list[dict] = []
    for m in messages:
        entry: dict = {"role": m.role, "text": m.text}
        if m.tool_calls:
            entry["tool_calls"] = [{"name": c.name, "arguments": c.arguments} for c in m.tool_calls]
        if m.role == "tool":
            entry["name"] = m.name
            entry["is_error"] = m.is_error
        if m.role == "assistant":
            entry["stop_reason"] = m.stop_reason
            if m.usage:
                entry["usage"] = {
                    "input_tokens": m.usage.input_tokens,
                    "output_tokens": m.usage.output_tokens,
                }
        out.append(entry)
    return out


def announced_then_stopped(stop_reason: str | None, passed: bool, final_text: str | None) -> bool:
    """The heuristic flag: ended without a tool call, did not pass, said it would act.

    `stop_reason == "done"` is the agent loop's word for "the final assistant message
    had no tool calls" (max_steps means the last message DID call a tool; error means
    there was no final message). The phrase test is the agent's own nudge trigger,
    announces_next_step, so the announced-then-stopped and nudges columns of the
    table count the same thing and can be read together. Round one's 7-8B swe
    failures all ended that way: "let me fix it and re-run", then no tool call. It is
    a flag to sort transcripts by, not a verdict.
    """
    if stop_reason != "done" or passed:
        return False
    return announces_next_step(final_text)


def run_once(
    model: str,
    task: Task,
    run_index: int,
    transcript_dir: Path,
    *,
    seed_base: int = DEFAULT_SEED_BASE,
    temperature: float = DEFAULT_TEMPERATURE,
    tool_capable: bool | None = None,
    extra_options: dict | None = None,
) -> dict:
    """Run `task` once on `model` in a fresh fixture copy; return the runs.jsonl record."""
    extra_options = dict(extra_options or {})
    # Temperature 0 is not determinism: Ollama still draws from its own random state in
    # places, so the seed is what pins a run. But a FIXED seed across the k runs would
    # make them near-identical repeats that measure nothing. Each run gets its own
    # seed, derived from its index, so the k runs are k different yet reproducible
    # draws — re-running the spike gives the same k, not a fresh roll of the dice.
    # (Round one showed that at temperature 0 even different seeds collapse to the
    # same trajectory on these prompts; that is what --temperature is for.)
    seed = seed_base + run_index
    # Built before the record, so the record's `options` is the dict this provider
    # sends: its frozen defaults, then the runner's settings, then --options.
    provider = make_provider(model, seed, temperature, extra_options)

    record: dict = {
        "model": model,
        "task": task.name,
        "run": run_index,
        "seed": seed,
        "temperature": temperature,
        # What the requests' `options` held and the --options dict by itself, so a
        # reader can tell an override from a default without diffing dicts.
        # temperature/seed above are the flag values; if --options overrode either,
        # `options` is the truth of what was sent.
        "options": provider.request_options(),
        "options_override": extra_options,
        "tool_capable": tool_capable,
        "passed": False,
        "verify_reason": None,
        "stop_reason": None,
        "steps": 0,
        "duration": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "n_tool_calls": 0,
        "n_tool_errors": 0,
        "n_soft_errors": 0,
        "n_distinct_tools": 0,
        "tools_used": [],
        "answered_without_tools": False,
        "announced_then_stopped": False,
        # How many times the agent had to nudge the model on ("you said you would
        # do that; go on") before it acted. Stays 0 for a run that raised, and is
        # absent from lines written before the field existed (read with .get).
        "nudges": 0,
        "last_stop_reason": None,
        "context_overflow": False,
        "final_text": "",
        # The Week 4c loop settings this run was made with (HARNESS_SETTINGS), so a
        # line from a rung that switches them on can never be confused with one from
        # the frozen rungs. Absent from lines written before the field existed,
        # which all ran with the loop's Week 4b behaviour, i.e. these very values.
        **HARNESS_SETTINGS,
        # The loop's own account of an unfinished run (RunResult.digest, Week 4c):
        # one line per tool call with its outcome. None for a finished run.
        "digest": None,
        "error": None,
        "placement": None,
        "transcript": None,
    }

    # A fresh copy of the pristine fixture, so nothing a previous run wrote survives.
    # __pycache__ is skipped in case someone ran pytest inside the fixture by hand.
    tmp = Path(tempfile.mkdtemp(prefix=f"spike-{task.name}-"))
    try:
        workspace_dir = tmp / "workspace"
        shutil.copytree(
            task.fixture,
            workspace_dir,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
        )
        workspace = Workspace(workspace_dir)

        agent = build_agent(provider, workspace)
        # The record says what the agent was actually built with, not what the
        # constant says it should have been.
        record.update(
            max_repeats=agent.max_repeats, wrap_up=agent.wrap_up, context_window=agent.context_window
        )

        started = time.perf_counter()
        result = None
        try:
            result = agent.run(task.prompt)
        except Exception as exc:  # noqa: BLE001 — a provider failure is data, not a crash
            record["error"] = f"{type(exc).__name__}: {exc}"[:500]
        record["duration"] = round(time.perf_counter() - started, 2)

        # Message-derived numbers first, then the RunResult's own where there is one.
        record.update(summarize_messages(agent.messages, num_ctx=record["options"]["num_ctx"]))

        # The whole conversation goes to its own file. A failed run's transcript is the
        # evidence of WHY it failed (a tool called with the wrong argument, an empty
        # reply right after a tool result), and it cannot be regenerated on demand: a
        # replay with the same seed does not reliably retrace the same trajectory.
        path = transcript_dir / f"{safe_name(model)}__{task.name}__run{run_index}.json"
        path.write_text(
            json.dumps(transcript(agent.messages), indent=2, default=str), encoding="utf-8"
        )
        record["transcript"] = str(path.relative_to(transcript_dir.parent))
        if result is None:
            record["stop_reason"] = "error"
            record["verify_reason"] = "not verified: the run raised before finishing"
            return record

        record.update(
            {
                "stop_reason": result.stop_reason,
                "steps": result.steps,
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "final_text": (result.text or "")[:300],
                "answered_without_tools": (
                    result.stop_reason == "done" and record["n_tool_calls"] == 0
                ),
                # getattr: the field is new in Week 4b, and a RunResult from an
                # older harness must not crash the runner.
                "nudges": getattr(result, "nudges", 0),
                "digest": getattr(result, "digest", None),
            }
        )

        # A max_steps run is verified too: the swe deliverable is the workspace, and a
        # model that fixed the code but never got round to saying so still fixed the
        # code (the text-based verifiers fail it on their own, since text is None).
        try:
            passed, reason = task.verify(str(workspace_dir), result.text or "")
        except Exception as exc:  # noqa: BLE001 — a verifier bug must show up as one
            passed, reason = False, f"verify.py crashed: {type(exc).__name__}: {exc}"
        record["passed"], record["verify_reason"] = bool(passed), reason
        # The heuristic reads the FULL final text, not the 300-char excerpt kept above.
        record["announced_then_stopped"] = announced_then_stopped(
            result.stop_reason, record["passed"], result.text
        )
        return record
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def format_line(r: dict) -> str:
    """The one stdout line per run."""
    status = "PASS" if r["passed"] else "FAIL"
    line = (
        f"[{r['model']}][{r['task']}][run {r['run']}] {status} "
        f"steps={r['steps']} secs={r['duration']:.1f} "
        f"tokens={r['input_tokens'] + r['output_tokens']}"
    )
    if r.get("tools_used"):
        line += " tools=" + ",".join(r["tools_used"])
    if r.get("context_overflow"):
        line += " CONTEXT-OVERFLOW"
    if r.get("announced_then_stopped"):
        line += " ANNOUNCED-THEN-STOPPED"
    if r.get("nudges"):
        line += f" nudges={r['nudges']}"
    if r.get("tool_capable") is False:
        line += " NO-TOOL-PROBE"
    if r["error"]:
        line += f" error={r['error'][:120]}"
    elif not r["passed"]:
        line += f" ({r['verify_reason']})"
    return line


# --- the report --------------------------------------------------------------------


def load_runs(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _ratio(num: int, den: int) -> str:
    return f"{num}/{den} ({100 * num / den:.0f}%)" if den else "n/a"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _md_cell(text: str, limit: int = 80) -> str:
    """One line of Markdown-table-safe text: pipes escaped, newlines collapsed."""
    flat = " ".join((text or "").split()).replace("|", "\\|")
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def probe_verdict(model: str, runs: list[dict], models_info: dict) -> bool | None:
    """The model's tool_capable flag: from models.json, else from any of its run lines.

    None means the model was never probed (round-one runs predate the probe).
    """
    verdict = models_info.get(model, {}).get("tool_capable")
    if verdict is None:
        verdict = next((r["tool_capable"] for r in runs if r.get("tool_capable") is not None), None)
    return verdict


def override_of(r: dict) -> dict:
    """The --options dict a run line was made with ({} for lines that predate the flag)."""
    return r.get("options_override") or {}


def row_key(r: dict) -> tuple[str, str]:
    """What a table row is keyed on: the model tag AND its --options override.

    A tag run with `--options '{"presence_penalty": 0}'` and the same tag run without
    are two different samplers, and averaging them into one row would hide exactly
    the difference the override was there to measure. The override is keyed by its
    canonical JSON so `{"a": 1, "b": 2}` and `{"b": 2, "a": 1}` are one row.
    """
    return (r["model"], json.dumps(override_of(r), sort_keys=True))


def row_label(model: str, override_json: str) -> str:
    """The model cell: the tag, plus the override when there is one.

    `qwen3.5:4b` -> "`qwen3.5:4b`"; with presence_penalty 0 forced ->
    "`qwen3.5:4b` (options: presence_penalty=0)".
    """
    override = json.loads(override_json)
    if not override:
        return f"`{model}`"
    settings = ", ".join(f"{k}={json.dumps(v)}" for k, v in override.items())
    return f"`{model}` (options: {settings})"


def build_table(runs: list[dict], models_info: dict, task_names: list[str]) -> str:
    """The results table, one row per (model, --options override), as GitHub Markdown."""
    keys = list(dict.fromkeys(row_key(r) for r in runs))  # first-seen order, no dupes
    # The data-via-tool column only means something when the data task was run.
    show_data_via_tool = "data" in task_names
    header = (
        ["model", "tool probe"]
        + task_names
        + (["data via tool"] if show_data_via_tool else [])
        + ["overall", "tool-error rate", "no-tool answers", "announced-then-stopped",
           "nudges", "errors", "ctx overflow", "avg steps", "avg tokens", "avg secs",
           "GPU %"]
    )
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    warnings: list[str] = []
    for key in keys:
        model, override_json = key
        mine = [r for r in runs if row_key(r) == key]
        cells = [row_label(model, override_json)]
        verdict = probe_verdict(model, mine, models_info)
        if verdict is None:
            cells.append("not probed")
        elif verdict:
            cells.append("ok")
        else:
            cells.append("**NO**")
            reply = models_info.get(model, {}).get("probe_reply") or ""
            error = models_info.get(model, {}).get("probe_error")
            detail = f"error: {error}" if error else f"reply began: \"{_md_cell(reply, 120)}\""
            warnings.append(
                f"> **Warning:** `{model}` emitted no tool call in the probe ({detail}). "
                "A model that cannot emit tool calls through Ollama is disqualified for "
                "this harness; its rows are kept for the record."
            )
        for task in task_names:
            of_task = [r for r in mine if r["task"] == task]
            cells.append(f"{sum(r['passed'] for r in of_task)}/{len(of_task)}")
        if show_data_via_tool:
            # Did the data task's arithmetic go through a compute tool at all? A pass
            # without one was arithmetic in prose, which the forty-row fixture is
            # meant to rule out; this column is where that would show.
            data_runs = [r for r in mine if r["task"] == "data"]
            computed = sum(1 for r in data_runs if COMPUTE_TOOLS & set(r.get("tools_used", [])))
            cells.append(f"{computed}/{len(data_runs)}" if data_runs else "n/a")
        cells.append(_ratio(sum(r["passed"] for r in mine), len(mine)))
        bad_calls = sum(r["n_tool_errors"] + r.get("n_soft_errors", 0) for r in mine)
        cells.append(_ratio(bad_calls, sum(r["n_tool_calls"] for r in mine)))
        cells.append(str(sum(r["answered_without_tools"] for r in mine)))
        cells.append(str(sum(1 for r in mine if r.get("announced_then_stopped"))))
        # Nudges summed over the row's runs; lines older than the field count 0.
        cells.append(str(sum(r.get("nudges", 0) for r in mine)))
        cells.append(str(sum(1 for r in mine if r["error"])))
        cells.append(str(sum(1 for r in mine if r.get("context_overflow"))))
        cells.append(f"{_mean([r['steps'] for r in mine]):.1f}")
        cells.append(f"{_mean([r['input_tokens'] + r['output_tokens'] for r in mine]):.0f}")
        cells.append(f"{_mean([r['duration'] for r in mine]):.1f}")
        # Placement is recorded on every run line, but the per-model summary is the
        # simplest place to read it back from.
        placement = models_info.get(model, {}).get("placement") or next(
            (r["placement"] for r in mine if r.get("placement")), None
        )
        pct = gpu_percent(placement)
        cells.append("?" if pct is None else str(pct))
        lines.append("| " + " | ".join(cells) + " |")
    if warnings:
        lines.append("")
        lines.extend(warnings)
    return "\n".join(lines)


def build_facts_table(runs: list[dict], models_info: dict) -> str:
    """Per-model facts: pull size, Modelfile PARAMETERs, placement, tokens/sec, the probe.

    One row per (model, override), like the results table: tokens/sec comes from the
    row's own runs, and the Modelfile column next to the override in the model cell
    is where a reader sees which baked-in setting the override replaced.
    """
    keys = list(dict.fromkeys(row_key(r) for r in runs))
    lines = [
        "| model | pull size | Modelfile PARAMETERs (ollama show --modelfile) | placement (ollama ps) "
        "| tokens/sec | tool probe | probe reply (raw, first 80 chars) |",
        "|---|---|---|---|---|---|---|",
    ]
    for key in keys:
        model, override_json = key
        mine = [r for r in runs if row_key(r) == key and r["duration"] > 0]
        info = models_info.get(model, {})
        params = info.get("modelfile_parameters")
        if params is None:
            params_cell = "?"  # never read (older models.json, or `ollama show` failed)
        elif not params:
            params_cell = "(none)"
        else:
            params_cell = _md_cell("; ".join(params), 120)
        # Output tokens over wall-clock seconds, summed over the model's runs. This is
        # the throughput the AGENT gets — prompt processing and tool time included —
        # not the raw decode speed, which is why it is lower than `ollama run`'s figure.
        seconds = sum(r["duration"] for r in mine)
        tps = sum(r["output_tokens"] for r in mine) / seconds if seconds else 0.0
        placement = info.get("placement") or next((r["placement"] for r in mine if r.get("placement")), None)
        verdict = probe_verdict(model, mine, models_info)
        probe = "not probed" if verdict is None else ("ok" if verdict else "**NO**")
        reply = info.get("probe_error") or info.get("probe_reply") or ""
        lines.append(
            f"| {row_label(model, override_json)} | {info.get('pull_size') or '?'} | {params_cell} "
            f"| {placement or '?'} | {tps:.1f} | {probe} | {_md_cell(reply) or '(empty)'} |"
        )
    return "\n".join(lines)


def reproduce_commands(
    runs: list[dict], task_names: list[str], k: int, temperature: float, seed_base: int, out_arg: str
) -> str:
    """One run_spike.py command per --options override present in runs.jsonl.

    A file can hold one tag run with `--options '{"presence_penalty": 0}'` and another
    run without, and one command line cannot reproduce both, so the models are grouped
    by override (first-seen order) and each group gets its own line.
    """
    groups: dict[str, list[str]] = {}
    for r in runs:
        models = groups.setdefault(json.dumps(override_of(r), sort_keys=True), [])
        if r["model"] not in models:
            models.append(r["model"])
    tasks_arg = ",".join(task_names)
    lines = []
    for override_json, models in groups.items():
        cmd = (
            f"python benchmark/spike/run_spike.py --models {','.join(models)} --tasks {tasks_arg} "
            f"--k {k} --temperature {temperature} --seed-base {seed_base}"
        )
        if override_json != "{}":
            cmd += f" --options '{override_json}'"
        lines.append(cmd + f" --out {out_arg}")
    return "\n".join(lines)


def build_report(
    runs: list[dict],
    models_info: dict,
    task_names: list[str],
    k: int,
    out: Path,
    temperature: float,
    seed_base: int,
) -> str:
    models = list(dict.fromkeys(r["model"] for r in runs))
    pulls = "\n".join(f"ollama pull {m}" for m in models)
    try:
        out_arg = str(out.resolve().relative_to(REPO_ROOT))
    except ValueError:
        out_arg = str(out)
    # Built outside the f-string below: 3.10/3.11 forbid reusing its quote inside.
    commands = reproduce_commands(runs, task_names, k, temperature, seed_base, out_arg)
    overrides = [k_ for k_ in dict.fromkeys(json.dumps(override_of(r), sort_keys=True) for r in runs) if k_ != "{}"]
    options_note = (
        " No --options override was used."
        if not overrides
        else " --options override(s) in this file: " + ", ".join(f"`{o}`" for o in overrides)
        + " — merged last into every request's `options`, so they win over the runner's own"
        " settings and over the tag's Modelfile PARAMETER lines (facts table); each"
        " run line records the merged dict (`options`) and the override (`options_override`),"
        " and a tag run with and without an override gets a row each."
    )
    # runs.jsonl appends, so a file can hold runs at more than one temperature (round
    # one's lines predate the field and mean 0.0). Say so rather than print one number.
    temps = sorted({r.get("temperature", 0.0) for r in runs})
    temp_note = (
        f"temperature {temperature}"
        if temps == [temperature]
        else f"temperature {temperature} for this invocation (runs.jsonl holds runs at "
        + ", ".join(str(t) for t in temps)
        + "; each line records its own)"
    )
    settings = textwrap.fill(
        f"Settings that make runs comparable: {temp_note}, seed {seed_base} + run index, "
        f"num_ctx {NUM_CTX}, num_predict {NUM_PREDICT} per step, max_steps {MAX_STEPS}, one "
        "HTTP attempt per call (no retries), one shared policy and tool set, think=False "
        f"for every Qwen 3 family tag, and the Week 4c loop behaviours off (max_repeats "
        f"{HARNESS_SETTINGS['max_repeats']}, wrap_up {HARNESS_SETTINGS['wrap_up']}, "
        f"context_window {HARNESS_SETTINGS['context_window']}; each line records its "
        f"own).{options_note} Runs append to `runs.jsonl`; delete the results directory "
        "for a clean slate.",
        width=80,
    )
    return f"""# Model spike results

Generated {datetime.now():%Y-%m-%d %H:%M} by `benchmark/spike/run_spike.py` from
{len(runs)} runs in `runs.jsonl` (one JSON line per run; the table is rebuilt from the
whole file each time the script runs). Each run's full conversation is in
`transcripts/`, which is where to look when a number here needs explaining.

{build_table(runs, models_info, task_names)}

Columns: tool probe is the pre-run check that the model can emit a tool call at all
(one tiny task with only get_current_time registered; `tool_capable` on every run
line, the raw reply in `models.json` and the facts table below; a **NO** disqualifies
the model for this harness and its rows are kept only for the record); each task
column is passes/runs for that task; data via tool (shown when the data task ran) is
data runs in which run_python or run_bash was invoked at all (a data pass without
one was arithmetic in prose; `tools_used` in runs.jsonl lists every run's tools);
overall is passes over all runs; tool-error rate is bad tool calls over all tool
calls, counting both kinds — hard errors, where the policy denied the call or the
tool raised (`n_tool_errors` in runs.jsonl), and soft errors, where the tool ran and
reported failure in its text, "Error: not a file" and the like (`n_soft_errors`);
no-tool answers is runs where the model answered without calling any tool (a
guess); announced-then-stopped is a HEURISTIC: runs that ended with a final message
carrying no tool call, did not pass, and whose final text ends on a line that
announces a step ("let me", "let's", "I will", "I'll", "I'm going to", "now I",
"next step") — the model announcing a next step and then ending its turn instead of
taking it (a phrase match, the agent's own nudge trigger, so read the transcript
before trusting any one count); nudges is the total number of times the agent
nudged the model on after such an announcement — the same phrase test, so the two
columns can be read together (`nudges` on each run line, 0 on lines that predate the
field; a run that passed after a nudge is a pass the harness bought); errors is
runs that died on a provider
error (an Ollama 500, a timeout; there are no retries, so one timeout is one error)
and count as fails; ctx overflow is runs in which some call's prompt plus output
reached num_ctx, so Ollama silently truncated the prompt and any failure after that
point is a budget problem rather than a tool-calling one; avg tokens is input +
output per run; GPU % is the share of the model that Ollama placed on the GPU. A
model cell with "(options: ...)" is a row run with that `--options` override (see
below); the same tag without it is a separate row.

## Model facts

{build_facts_table(runs, models_info)}

Modelfile PARAMETERs are the sampling settings the tag itself bakes in (`ollama show
<tag> --modelfile`; `modelfile_parameters` in `models.json`) and applies to every
request that does not set the same key; a request option — the runner's temperature,
or anything in `--options` — overrides the PARAMETER of the same name.

## How to reproduce

```bash
cd <repo root>
source .venv/bin/activate          # pytest must be installed (the swe tasks run it)
{pulls}
{commands}
```

{settings}
"""


# --- main --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the model spike: k runs of each task on each model, graded by verify.py."
    )
    parser.add_argument("--models", default=DEFAULT_MODELS, help="comma-separated Ollama tags")
    parser.add_argument("--tasks", default=None, help="comma-separated task names (default: all)")
    parser.add_argument("--k", type=int, default=3, help="runs per model per task")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="results directory")
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help="sampling temperature passed to Ollama (default 0.0; round two used 0.3)",
    )
    parser.add_argument(
        "--seed-base", type=int, default=DEFAULT_SEED_BASE,
        help="run i of each task uses seed SEED_BASE + i (default 42)",
    )
    parser.add_argument(
        "--options", default=None, metavar="JSON",
        help=(
            "a JSON object merged into every request's Ollama \"options\" AFTER the "
            "runner's own temperature/num_ctx/seed/num_predict, so it can override them "
            "and the tag's Modelfile PARAMETER lines; applied to every model of this "
            "invocation and recorded on every run line. Example: '{\"presence_penalty\": 0}'"
        ),
    )
    args = parser.parse_args(argv)
    extra_options = parse_options(args.options)
    if extra_options:
        print(
            f"--options {json.dumps(extra_options)} merged into every request's options "
            "(after the runner's own; overrides Modelfile PARAMETERs)",
            flush=True,
        )

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = discover_tasks([t.strip() for t in args.tasks.split(",")] if args.tasks else None)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    runs_path = out / "runs.jsonl"
    models_path = out / "models.json"
    transcript_dir = out / "transcripts"
    transcript_dir.mkdir(exist_ok=True)
    models_info: dict = json.loads(models_path.read_text()) if models_path.is_file() else {}

    with runs_path.open("a", encoding="utf-8") as runs_file:
        for model in models:
            info = models_info.setdefault(model, {})
            info["pull_size"] = ollama_column("list", model, "SIZE") or info.get("pull_size")
            # The sampling settings the tag bakes in. Read before any request, so a row
            # can be read next to them; None (the command failed) keeps an older value.
            params = modelfile_parameters(model)
            if params is not None:
                info["modelfile_parameters"] = params
            shown = info.get("modelfile_parameters")
            print(
                f"[{model}] Modelfile PARAMETERs: "
                + ("?" if shown is None else ("; ".join(shown) or "(none)")),
                flush=True,
            )

            failure = warm_up(model, args.seed_base, args.temperature, extra_options)
            if failure:
                print(f"[{model}] warm-up failed: {failure}", file=sys.stderr, flush=True)
            # Where Ollama put the model (all on the GPU, or split with the CPU). Read
            # once per model, right after its first call, while it is certainly loaded.
            placement = ollama_column("ps", model, "PROCESSOR")

            # The tool-call probe, before any task: can this model call a tool at all?
            probe = probe_tool_calling(
                model, args.seed_base, args.temperature, transcript_dir, extra_options
            )
            info.update(probe)
            if probe["tool_capable"]:
                print(f"[{model}] tool probe: ok (a tool call was emitted)", flush=True)
            else:
                why = probe["probe_error"] or f"reply began: {probe['probe_reply']!r}"
                print(
                    f"[{model}] WARNING tool probe: NO tool call emitted ({why}); "
                    "the model is disqualified for this harness, running its tasks for the record",
                    file=sys.stderr,
                    flush=True,
                )
            models_path.write_text(json.dumps(models_info, indent=2) + "\n", encoding="utf-8")

            for task in tasks:
                for i in range(args.k):
                    record = run_once(
                        model, task, i, transcript_dir,
                        seed_base=args.seed_base,
                        temperature=args.temperature,
                        tool_capable=probe["tool_capable"],
                        extra_options=extra_options,
                    )
                    if placement is None:  # the warm-up failed; the model may be loaded now
                        placement = ollama_column("ps", model, "PROCESSOR")
                    record["placement"] = placement
                    # Written and flushed per run, so an interrupted spike keeps its data.
                    runs_file.write(json.dumps(record) + "\n")
                    runs_file.flush()
                    print(format_line(record), flush=True)

            info["placement"] = placement
            models_path.write_text(json.dumps(models_info, indent=2) + "\n", encoding="utf-8")

    all_runs = load_runs(runs_path)
    task_names = sorted({r["task"] for r in all_runs}, key=_task_sort_key)
    report = build_report(
        all_runs, models_info, task_names, args.k, out, args.temperature, args.seed_base
    )
    (out / "results.md").write_text(report, encoding="utf-8")
    print()
    print(build_table(all_runs, models_info, task_names))
    print(f"\nwrote {out / 'results.md'}")


if __name__ == "__main__":
    main()
