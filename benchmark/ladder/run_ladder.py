#!/usr/bin/env python
"""The capability ladder runner: one frozen model, six levels of help, one table.

The experiment (docs/plan.md, "the ladder")
-------------------------------------------
Take three small broken programs (benchmark/spike/tasks/swe, swe2, swe3: a pytest
fails, the fix belongs in the source, the test file must not change) and ONE small
model, frozen after the spike: qwen3.5:4b on Ollama, temperature 0.3, seed 42 + run,
num_ctx 8192, num_predict 2048, presence_penalty 0, thinking off — round three's exact
settings. Ask the model to fix each program k=3 times at each of six levels. Every
level gives the model exactly one thing more than the level below:

    0 bare       the question and the code, one model call, no system prompt
    1 prompted   + a system prompt (the persona and the list of files)
    2 one_tool   + one tool, run_bash: the model can act, not only answer
    3 tools      + the full read/write tool set (the round-two harness)
    4 harness    + the loop guards of Weeks 4b/4c (edit_file, syntax gate, nudge, ...)
    5 agents     + sub-agents

Because each step adds one thing, a jump in the pass rate between two adjacent levels
can be credited to that one thing and nothing else. Round three showed the size of the
prize across three small models: 12/27 -> 26/27 when the harness grew up
(docs/observations.md: qwen3:4b-instruct 4/9 -> 9/9, qwen3:8b 1/9 -> 8/9, qwen3.5:4b
9/9 -> 9/9). The frozen qwen3.5:4b was 9/9 on both harnesses, which is why the ladder
starts further down, with no tools at all: the room to climb is below level 3. The
table this script writes — rows are levels, the "overall" column is the number that
climbs — is the result the whole project exists to produce.

What this file does and does not decide
---------------------------------------
The LEVELS live in benchmark/ladder/levels.py: what each one adds, the exact prompt,
which tools, which loop settings. This runner never looks inside a level. It gives
each level a provider, a fresh copy of the fixture, the task prompt and a transcript
path, and grades whatever is left in the workspace afterwards with the task's own
verify.py — the same call the spike makes. So the runner is the part of the experiment
that must be IDENTICAL for every level: same fixture copy, same grading, same seeds,
same environment. The one thing that differs between rows is the level.

What is reused from the spike, and why nothing is copied
--------------------------------------------------------
benchmark/spike/run_spike.py already has the task loader (task.md + verify.py), the
frozen Ollama provider with its option layering (SpikeProvider, make_provider,
runner_options), the file-name sanitiser and the runs.jsonl reader. They are loaded
from that file with importlib (tests/conftest.py loads it the same way) rather than
copied: a copy would drift, and a ladder run must be built on the SAME provider
settings the frozen model was chosen with, or the rows cannot be compared with round
three. For `--provider openai` (a 27B model on vLLM, later) the OpenAI-compatible
provider is built here with every frozen setting a chat-completions request can
carry: temperature, seed, the per-step output cap (max_tokens = the spike's
num_predict) and, for a Qwen 3 family model, thinking switched off the way vLLM
takes it (chat_template_kwargs.enable_thinking = false). What it cannot carry is
num_ctx: the context length is the server's --max-model-len, so the runner tells the
loop the same 8192 for context clearing and the README tells the reader to serve the
model at that length.

Why `python` must be put on PATH by the runner
----------------------------------------------
Every task prompt says "run the tests with pytest", and from level 2 up the model does
that through run_bash — typically `python -m pytest -q`. run_bash starts a plain shell
with the process's environment, so `python` is whatever the shell finds on PATH. The
spike rounds were run from an activated venv (`source .venv/bin/activate` is in every
results.md), which put .venv/bin first on PATH; the ladder should not depend on the
reader remembering that. On this machine, without activation, `python` is not on PATH
at all (`sh: python: not found`; when WSL's Windows-interop entries do match, the hit
cannot be executed, "Permission denied") and the `pytest` the shell finds belongs to a
different interpreter (~/.local/bin). Every tool level would then
fail its first test run for a reason that has nothing to do with the model, and level
0/1 (no tools) would be unaffected: a bias in the very comparison the ladder makes.
So run_matrix() prepends the running interpreter's bin directory to PATH before the
first run (put_interpreter_first_on_path), once, for every level alike. verify.py is
not affected either way — it runs pytest with sys.executable — which is why the
verifier's verdict was always sound and only the model's own test runs were at risk.

Output
------
<out>/runs.jsonl        one JSON object per run, appended and flushed as each run ends,
                        so an interrupted matrix keeps every finished row
<out>/transcripts/      L<level>-<task>-r<run>.jsonl, the full conversation of that run
<out>/results.md        the table, rebuilt from the whole runs.jsonl at the end

Runs go level-major (level 0, all tasks, all runs; then level 1; ...), so a matrix
that is stopped early still has complete rows for the levels it finished. `--resume`
skips every (level, task, run) already in runs.jsonl, so a stopped matrix can be
continued and a new level added later. `--report-only DIR` rebuilds results.md from an
existing runs.jsonl without running anything.

    python benchmark/ladder/run_ladder.py                       # 6 levels x 3 tasks x k=3
    python benchmark/ladder/run_ladder.py --levels 0,2,5 --tasks swe --k 1
    python benchmark/ladder/run_ladder.py --resume --out benchmark/ladder/results/<dir>
    python benchmark/ladder/run_ladder.py --report-only benchmark/ladder/results/<dir>
    python benchmark/ladder/run_ladder.py --provider openai --base-url http://host:8000/v1 \\
        --model Qwen/Qwen3-27B                                  # the same ladder, a 27B
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import textwrap
import time
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterable, Sequence

LADDER_DIR = Path(__file__).resolve().parent
REPO_ROOT = LADDER_DIR.parent.parent
RUN_SPIKE = REPO_ROOT / "benchmark" / "spike" / "run_spike.py"
LEVELS_PY = LADDER_DIR / "levels.py"
DEFAULT_RESULTS_ROOT = LADDER_DIR / "results"

# `python benchmark/ladder/run_ladder.py` puts benchmark/ladder on sys.path, not the
# repo root, and cornac is not pip-installed in the dev venv (the spike does the same).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cornac.core.messages import Message, Usage  # noqa: E402
from cornac.providers.base import Provider  # noqa: E402
from cornac.tools.workspace import Workspace  # noqa: E402


def load_module(path: Path, name: str) -> ModuleType:
    """Load a script as a module by file path, registered in sys.modules under `name`.

    Both run_spike.py and levels.py are scripts outside the package, so they cannot
    be imported by name from here. Registering the module before executing it
    matters: both use `from __future__ import annotations` with @dataclass, and the
    dataclass machinery resolves a class's annotation strings through
    sys.modules[cls.__module__] — an unregistered module makes that lookup fail.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# The spike runner, loaded once at import. Everything the ladder needs from it is
# reached as `spike.<name>`: load_task/discover_tasks/TASKS_DIR, make_provider and
# SpikeProvider, runner_options, safe_name, load_runs, and the frozen constants
# (DEFAULT_MODELS, MAX_STEPS, NUM_CTX, NUM_PREDICT, DEFAULT_SEED_BASE, REQUEST_TIMEOUT).
spike = load_module(RUN_SPIKE, "run_spike_for_ladder")

# The ladder's own defaults. Temperature is the one number that differs from the
# spike's DEFAULT_TEMPERATURE (0.0): round one showed that at 0 the three seeds
# collapse to byte-identical runs and k=3 measures nothing, so rounds two and three
# sampled at 0.3 with seeds 42/43/44, and the ladder keeps that.
DEFAULT_LEVELS = "0-5"
DEFAULT_TASKS = ("swe", "swe2", "swe3")
DEFAULT_K = 3
DEFAULT_TEMPERATURE = 0.3
DEFAULT_SEED_BASE = spike.DEFAULT_SEED_BASE   # 42
DEFAULT_MODEL = spike.DEFAULT_MODELS          # qwen3.5:4b, the frozen tag
DEFAULT_MAX_STEPS = spike.MAX_STEPS           # 12
# How much of the final reply goes on the run line. 2000 characters is enough to
# read what the model said it fixed; the whole text is in the transcript.
FINAL_TEXT_CHARS = 2000
# A run's stop reasons in the order the table lists them; anything else follows.
STOP_REASON_ORDER = ("done", "max_steps", "stuck", "error")

# --- infrastructure retries -------------------------------------------------------------
# The first full matrix (2026-09-27, 54 runs) lost three cells to the same fault: Ollama's
# llama-server process died ("model runner has unexpectedly stopped"; the daemon log shows
# a GPU-discovery timeout while it reloaded) and the run's very first request came back
# as a 500. That is the machine failing, not the model, and a table that counts it as a
# model failure lies by one cell. So a run whose error carries one of these signatures is
# re-run from scratch — fresh fixture copy, same seed — after a pause long enough for the
# daemon to reload the model, up to INFRA_RETRIES times. The run line keeps the count and
# every error text, so a reader can see that a retry happened and why. The signatures are
# deliberately narrow: a 500 for any other reason (a context overflow, a bad request) is
# NOT retried, because that one may well be the harness's or the model's doing.
INFRA_ERROR_MARKERS = (
    "model runner has unexpectedly stopped",
    "ConnectError",
    "RemoteProtocolError",
)
INFRA_RETRIES = 2
INFRA_WAIT_SECONDS = 10.0


def is_infrastructure_error(error: str | None) -> bool:
    """True if a run's error text is the daemon dying, not the model or the harness failing."""
    return bool(error) and any(marker in error for marker in INFRA_ERROR_MARKERS)
# How the table labels them. The loop's word for "the model stopped on its own" is
# "done" (RunResult.stop_reason), and a reader skimming the table takes "done 9" next
# to "0/9" as nine finished jobs. The run lines keep the loop's word; the table says
# what happened: the model answered (rightly or wrongly — the pass columns say which).
STOP_REASON_LABELS = {"done": "answered"}

# The keys every runs.jsonl line carries, in this order (the contract's list). A
# reader who greps the file can rely on all of them being present on every line.
RUN_KEYS: tuple[str, ...] = (
    "level", "level_key", "adds", "task", "run", "seed", "temperature", "model",
    "provider", "passed", "verify_note", "stop_reason", "steps", "prompt_tokens",
    "output_tokens", "total_tokens", "seconds", "tool_calls", "tool_errors", "nudges",
    "children", "files_applied", "rejected", "digest", "summary", "final_text",
    "transcript", "started_at",
)

ProviderFactory = Callable[[int], Provider]


# --- the environment ----------------------------------------------------------------


def put_interpreter_first_on_path() -> str:
    """Prepend the running interpreter's bin directory to PATH; return that directory.

    See the module docstring ("Why `python` must be put on PATH by the runner"). The
    directory is taken from sys.executable WITHOUT resolving symlinks: in a uv venv
    `.venv/bin/python` is a link into uv's own Python install, whose bin directory
    has no pytest; it is the venv's bin directory, where pytest and the `python`
    name both live, that must come first. Idempotent, so calling it per matrix (or
    per test) never grows PATH; identical for every level, because it runs before
    the first run and is never undone.
    """
    bin_dir = str(Path(sys.executable).parent)
    parts = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    if not parts or parts[0] != bin_dir:
        os.environ["PATH"] = os.pathsep.join([bin_dir] + [p for p in parts if p != bin_dir])
    return bin_dir


# --- levels ---------------------------------------------------------------------------


def load_levels(path: Path | None = None) -> ModuleType:
    """benchmark/ladder/levels.py as a module (LEVELS, and the level constants).

    The path is read from LEVELS_PY at call time, not bound as a default, so a test
    can point the runner at a stub levels file and drive main() end to end without
    the real levels (and without a model).
    """
    return load_module(path if path is not None else LEVELS_PY, "ladder_levels")


def parse_levels(text: str) -> list[int]:
    """"0-5" -> [0..5], "0,2,5" -> [0, 2, 5], "3" -> [3]; ranges and lists may mix.

    Sorted and de-duplicated: the ladder is always run bottom-up, whatever order the
    flag names the levels in, so the rows of a partial run are the lowest levels.
    """
    numbers: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                lo, hi = (int(x) for x in part.split("-", 1))
                if hi < lo:
                    raise ValueError
                numbers.update(range(lo, hi + 1))
            else:
                numbers.add(int(part))
        except ValueError:
            raise SystemExit(
                f"--levels: cannot read {part!r}; use a range like 0-5 or a list like 0,2,5"
            ) from None
    if not numbers:
        raise SystemExit("--levels names no level")
    return sorted(numbers)


def select_levels(levels: Sequence, numbers: Iterable[int]) -> list:
    """The Level objects with these numbers, in ladder order; unknown numbers are an error."""
    wanted = set(numbers)
    known = {level.number for level in levels}
    missing = sorted(wanted - known)
    if missing:
        raise SystemExit(f"unknown level(s) {missing}; levels.py defines {sorted(known)}")
    return [level for level in levels if level.number in wanted]


# --- providers ------------------------------------------------------------------------


def make_provider_factory(
    kind: str, model: str, temperature: float, base_url: str | None = None
) -> ProviderFactory:
    """A function seed -> Provider, for the two backends the ladder runs on.

    A NEW provider per run, built from the run's seed, is how the spike does it too:
    the seed is baked into the provider's options, and a run line records what the
    provider actually sent. `ollama` reuses the spike's frozen provider unchanged
    (SpikeProvider via make_provider: presence_penalty 0, think off, num_ctx 8192,
    num_predict 2048, one HTTP attempt). `openai` is the same experiment pointed at a
    vLLM or llama.cpp server, with the same settings in that server's words
    (openai_options): temperature and seed; max_tokens = NUM_PREDICT, the per-step
    output cap, without which a rambling 27B step runs to the server's limit while a
    4B step is cut at 2048 and recorded as "length"; and thinking off for a Qwen 3
    family model, or the table would compare a thinking 27B against a non-thinking
    4B, with the chain of thought inflating tokens and seconds. The context window
    is set to the same 8192 so a level whose clearing is sized by the provider
    (levels 4-5) behaves the same on both backends (the server's own context length
    is not a request field: serve the model at 8192, see the README); one attempt
    and the spike's timeout, for the reasons make_provider gives (a retried timeout
    would fold minutes into `seconds` with nothing on record).
    """
    if kind == "ollama":
        return lambda seed: spike.make_provider(model, seed, temperature)
    if kind == "openai":
        from cornac.providers.openai_compat import DEFAULT_BASE_URL, OpenAICompatibleProvider

        url = base_url or DEFAULT_BASE_URL
        return lambda seed: OpenAICompatibleProvider(
            url,
            model,
            options=openai_options(model, seed, temperature),
            context_window=spike.NUM_CTX,
            timeout=spike.REQUEST_TIMEOUT,
            retries=0,
        )
    raise SystemExit(f"--provider must be ollama or openai, got {kind!r}")


def openai_options(model: str, seed: int, temperature: float) -> dict:
    """The frozen generation settings as chat-completions request fields.

    The Ollama side sends num_predict, presence_penalty 0 and think=False (SpikeProvider);
    the equivalents here are max_tokens, nothing (the OpenAI default presence_penalty
    is 0) and vLLM's chat_template_kwargs switch, sent only for a tag whose lowercase
    name contains "qwen3" — the same rule OllamaProvider.think_flag applies, because a
    server whose template has no thinking mode may reject the field. Every key here is
    recorded on the run line (the provider's request_options), so the report's header
    says what was actually sent.
    """
    options: dict = {"temperature": temperature, "seed": seed, "max_tokens": spike.NUM_PREDICT}
    if "qwen3" in model.lower():
        options["chat_template_kwargs"] = {"enable_thinking": False}
    return options


def warm_up(provider: Provider) -> str | None:
    """One tiny request so the first real run does not pay for loading the model.

    The spike's warm_up builds an Ollama provider of its own; this one takes any
    provider (the factory's), so the OpenAI path is warmed the same way. Returns the
    error text on failure — a model that is not pulled, a server that is down —
    which is reported once before it would fail every run.
    """
    try:
        provider.complete([Message.user("Reply with the single word: ready")], [])
    except Exception as exc:  # noqa: BLE001 — reported, not fatal
        return f"{type(exc).__name__}: {exc}"
    return None


# --- runs.jsonl -----------------------------------------------------------------------


def read_runs(path: Path, *, repair: bool = False, warn: Callable[[str], None] | None = None) -> list[dict]:
    """The runs in a runs.jsonl, tolerating a torn LAST line and nothing else.

    Every record is written with write()+flush() — one write(2) call — so Ctrl-C
    leaves clean lines; but a SIGKILL or OOM kill landing inside that call, a disk
    that fills mid-record, or a host crash before the page cache reached the disk can
    leave a partial final line, and a 54-run matrix on a laptop runs for hours. The
    spike's load_runs raises JSONDecodeError on any bad line, and that would defeat
    the one flag written for a crash: --resume could not read the finished rows to
    skip them, and --report-only could not rebuild the table from them. So the file
    is read line by line here. A malformed LAST line is dropped with a warning that
    names it (to stderr by default), and --resume then re-runs that one triple; with
    `repair` the file is also cut back to the end of the last good line, so the next
    record appended starts on a line of its own instead of gluing itself to the torn
    one. A malformed line anywhere ELSE is not a crash artefact but a damaged file,
    and is a hard error naming the line, because a table built around it would be
    built from a file nobody can trust.
    """
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    numbered = [(i, line) for i, line in enumerate(lines, start=1) if line.strip()]
    runs: list[dict] = []
    for position, (lineno, line) in enumerate(numbered):
        try:
            runs.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if position != len(numbered) - 1:
                raise ValueError(
                    f"{path}: line {lineno} is not valid JSON ({exc.msg}); only a torn "
                    "last line is repaired, a damaged line elsewhere needs a human"
                ) from exc
            (warn or _stderr)(
                f"warning: {path}: dropping the incomplete last line ({lineno}); the run "
                "it belonged to was interrupted mid-write and will be re-run under --resume"
            )
            if repair:
                keep = "".join(lines[: lineno - 1])
                path.write_text(keep, encoding="utf-8")
    return runs


def _stderr(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def _stdout(text: str) -> None:
    """One line to stdout, flushed: run_matrix's default `echo`.

    A matrix is run as `python run_ladder.py > log 2>&1` and tailed from another
    terminal, and a redirected stdout is block-buffered: a plain print sits in the
    buffer until it fills or the process exits. The first smoke run of the ladder
    (2026-09-27) showed nothing in its log for six runs and then all six lines at
    once, when it ended; a 54-run matrix that takes hours would look hung the whole
    time. The run line is the only sign of life while a level works, so it is
    flushed as the run record itself is.
    """
    print(text, flush=True)


# --- one run ----------------------------------------------------------------------------


def _ignore_caches(*_) -> set[str]:
    """shutil.copytree ignore function: never copy Python's caches into a fixture copy."""
    return {"__pycache__", ".pytest_cache"}


def transcript_name(level_number: int, task_name: str, run_index: int) -> str:
    """transcripts/L<level>-<task>-r<run>.jsonl (the task name sanitised like a model tag)."""
    return f"transcripts/L{level_number}-{spike.safe_name(task_name)}-r{run_index}.jsonl"


def run_one(
    level,
    task,
    run_index: int,
    provider: Provider,
    out: Path,
    *,
    seed: int,
    temperature: float,
    max_steps: int,
    model: str | None = None,
    provider_label: str | None = None,
) -> dict:
    """Run `task` once at `level` in a fresh fixture copy, grade it, return the run line.

    The order of events is the experiment's fixed procedure: copy the pristine
    fixture into a temp directory (a run can never see what an earlier run wrote),
    hand the level a Workspace on that copy, time level.run() with the runner's own
    clock — the same clock at every level, whether the level made one model call or
    ran a twelve-step agent — then call the task's verify.py on the copy exactly as
    the spike does. A level that raises (an Ollama 500, a timeout) is a FAILED run
    with stop_reason "error" and the exception text on the line, not a crash: a
    matrix that dies on run 40 of 54 wastes the first 39. The workspace is graded
    even then, because the deliverable is the workspace, and a level that crashed
    after writing a correct fix still wrote it.
    """
    started_at = datetime.now().isoformat(timespec="seconds")
    rel = transcript_name(level.number, task.name, run_index)
    transcript_path = out / rel
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    # TranscriptWriter appends. A transcript left by a run that was interrupted before
    # its line reached runs.jsonl would otherwise get a second run's events appended
    # to it under --resume; the orphan is removed so the file is that run's alone.
    if transcript_path.exists():
        transcript_path.unlink()

    record: dict = {
        "level": level.number,
        "level_key": level.key,
        "adds": level.adds,
        "task": task.name,
        "run": run_index,
        "seed": seed,
        "temperature": temperature,
        "model": model if model is not None else getattr(provider, "model", None),
        "provider": provider_label if provider_label is not None else provider.name,
        "passed": False,
        "verify_note": None,
        "stop_reason": None,
        "steps": 0,
        "prompt_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "seconds": 0.0,
        "tool_calls": 0,
        "tool_errors": 0,
        "nudges": 0,
        "children": 0,
        "files_applied": [],
        "rejected": 0,
        "digest": None,
        "summary": None,
        "final_text": "",
        "transcript": rel,
        "started_at": started_at,
        # Beyond the contract's keys: the fixed conditions a reader needs next to a
        # row (what the provider actually sent, the step budget), the level's own
        # word on whether it produced output at all, and the error text of a run
        # that raised. All are None/absent-safe for older lines.
        "max_steps": max_steps,
        "options": _request_options(provider),
        "passed_precheck": None,
        "error": None,
        # Set by run_matrix when the cell had to be re-run after a daemon crash.
        "infra_retries": 0,
        "infra_errors": [],
    }

    tmp = Path(tempfile.mkdtemp(prefix=f"ladder-L{level.number}-{task.name}-"))
    try:
        workspace_dir = tmp / "workspace"
        shutil.copytree(task.fixture, workspace_dir, ignore=_ignore_caches)
        workspace = Workspace(workspace_dir)

        started = time.perf_counter()
        outcome = None
        try:
            outcome = level.run(provider, workspace, task.prompt, transcript_path)
        except Exception as exc:  # noqa: BLE001 — a provider failure is data, not a crash
            record["error"] = f"{type(exc).__name__}: {exc}"[:500]
        record["seconds"] = round(time.perf_counter() - started, 2)

        final_text = ""
        if outcome is None:
            record["stop_reason"] = "error"
        else:
            final_text = getattr(outcome, "final_text", None) or ""
            usage = getattr(outcome, "usage", None) or Usage()
            record.update(
                {
                    "passed_precheck": bool(getattr(outcome, "passed_precheck", True)),
                    "stop_reason": getattr(outcome, "stop_reason", None),
                    "steps": int(getattr(outcome, "steps", 0) or 0),
                    "prompt_tokens": usage.prompt_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                    "tool_calls": int(getattr(outcome, "tool_calls", 0) or 0),
                    "tool_errors": int(getattr(outcome, "tool_errors", 0) or 0),
                    "nudges": int(getattr(outcome, "nudges", 0) or 0),
                    "children": int(getattr(outcome, "children", 0) or 0),
                    "files_applied": list(getattr(outcome, "files_applied", None) or []),
                    "rejected": int(getattr(outcome, "rejected", 0) or 0),
                    "digest": getattr(outcome, "digest", None),
                    "summary": getattr(outcome, "summary", None),
                    "final_text": final_text[:FINAL_TEXT_CHARS],
                }
            )

        # Grading: the same call, with the same arguments, as run_spike.run_once. The
        # FULL final text goes to the verifier (the swe verifiers ignore it; the
        # spike's text-based ones do not), the run line keeps the first 2000 chars.
        try:
            passed, note = task.verify(str(workspace_dir), final_text)
        except Exception as exc:  # noqa: BLE001 — a verifier bug must show up as one
            passed, note = False, f"verify.py crashed: {type(exc).__name__}: {exc}"
        record["passed"], record["verify_note"] = bool(passed), note
        return record
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _request_options(provider: Provider) -> dict | None:
    """What the provider sends as generation settings, if it can say (else None)."""
    getter = getattr(provider, "request_options", None)
    if getter is None:
        return None
    try:
        options = getter()
    except Exception:  # noqa: BLE001 — a record field, never a reason to fail a run
        return None
    return dict(options) if isinstance(options, dict) else None


def format_line(r: dict) -> str:
    """The one stdout line per run: `L2 one_tool  swe  r1  PASS  6 steps  14,231 tok  21.3 s`."""
    status = "PASS" if r["passed"] else "FAIL"
    line = (
        f"L{r['level']} {r['level_key']:<9} {r['task']:<5} r{r['run']}  {status}  "
        f"{r['steps']} steps  {r['total_tokens']:,} tok  {r['seconds']:.1f} s"
    )
    if r.get("error"):
        line += f"  error={r['error'][:120]}"
    elif not r["passed"] and r.get("verify_note"):
        line += f"  ({r['verify_note']})"
    return line


# --- the matrix -------------------------------------------------------------------------


def run_matrix(
    levels: Sequence,
    tasks: Sequence,
    k: int,
    provider_factory: ProviderFactory,
    out: str | Path,
    *,
    seed_base: int = DEFAULT_SEED_BASE,
    temperature: float = DEFAULT_TEMPERATURE,
    max_steps: int = DEFAULT_MAX_STEPS,
    model: str | None = None,
    provider_label: str | None = None,
    resume: bool = False,
    echo: Callable[[str], None] = _stdout,
    infra_retries: int = INFRA_RETRIES,
    infra_wait: float = INFRA_WAIT_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict]:
    """Run every (level, task, run) and return ALL the runs in <out>/runs.jsonl.

    This is the testable core: main() only parses flags and calls it. Levels come
    from levels.py (or a stub of the same shape in a test), tasks from the spike's
    discover_tasks, and `provider_factory(seed)` makes the provider for one run — a
    scripted one in tests, the frozen Ollama one in a real matrix.

    Order is level-major, then task, then run, so stopping early leaves whole rows.
    Run i of every (level, task) uses seed `seed_base + i`: the same k draws at every
    level (a level is not allowed a luckier seed than its neighbour), reproducible on
    a re-run, and different from each other (round one showed a fixed seed makes k
    runs near-identical repeats). Each line is written and flushed as its run ends,
    and `echo` (the stdout line per run) flushes too, so a redirected log shows the
    run the moment it finishes (_stdout).

    `resume=True` skips the triples already in runs.jsonl. Without it, a runs.jsonl
    that already exists is refused rather than appended to: the table is built from
    the whole file, and a duplicated row would count twice.

    A run that fails with an infrastructure error (is_infrastructure_error) is re-run
    from scratch after `infra_wait` seconds, up to `infra_retries` times; only the final
    attempt is written, carrying `infra_retries` and every `infra_errors` text. `sleep`
    is injectable so a test does not wait.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "transcripts").mkdir(exist_ok=True)
    runs_path = out / "runs.jsonl"
    if runs_path.exists() and not resume:
        raise FileExistsError(
            f"{runs_path} already holds runs; pass resume=True (--resume) to continue it, "
            "or choose a fresh out directory"
        )
    put_interpreter_first_on_path()  # before the first run, once, for every level

    # A torn last line (a kill mid-write) is cut off here, so the append below starts
    # on a fresh line and that triple is simply not in `done`.
    done = {(r["level"], r["task"], r["run"]) for r in read_runs(runs_path, repair=True)}
    with runs_path.open("a", encoding="utf-8") as runs_file:
        for level in levels:
            for task in tasks:
                for i in range(k):
                    if (level.number, task.name, i) in done:
                        continue
                    seed = seed_base + i
                    errors_seen: list[str] = []
                    for attempt in range(infra_retries + 1):
                        provider = provider_factory(seed)
                        record = run_one(
                            level, task, i, provider, out,
                            seed=seed,
                            temperature=temperature,
                            max_steps=max_steps,
                            model=model,
                            provider_label=provider_label,
                        )
                        if not (is_infrastructure_error(record.get("error")) and attempt < infra_retries):
                            break
                        errors_seen.append(record["error"])
                        echo(
                            f"   infrastructure error, not the model's: {record['error'][:90]} "
                            f"-> waiting {infra_wait:g}s, re-running from scratch "
                            f"({attempt + 1}/{infra_retries})"
                        )
                        sleep(infra_wait)
                    record["infra_retries"] = len(errors_seen)
                    record["infra_errors"] = errors_seen
                    runs_file.write(json.dumps(record) + "\n")
                    runs_file.flush()
                    echo(format_line(record))
    return read_runs(runs_path)


# --- the report -------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ratio(num: int, den: int) -> str:
    return f"{num}/{den} ({100 * num / den:.0f}%)" if den else "n/a"


def _md_cell(text, limit: int = 90) -> str:
    """One line of Markdown-table-safe text: pipes escaped, newlines collapsed, capped."""
    flat = " ".join(str(text or "").split()).replace("|", "\\|")
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def level_rows(runs: list[dict]) -> list[tuple[int, str, str]]:
    """(number, key, adds) for every level present, in ladder order."""
    seen: dict[int, tuple[int, str, str]] = {}
    for r in runs:
        seen.setdefault(r["level"], (r["level"], r.get("level_key", "?"), r.get("adds", "")))
    return [seen[n] for n in sorted(seen)]


def task_names(runs: list[dict]) -> list[str]:
    """The tasks present, in first-seen order (the order the matrix ran them)."""
    return list(dict.fromkeys(r["task"] for r in runs))


def stop_reasons(rows: list[dict]) -> str:
    """"answered 8 · max_steps 1": each stop reason with its count, known ones first.

    The loop's "done" is shown under its table label (STOP_REASON_LABELS)."""
    counts: dict[str, int] = {}
    for r in rows:
        reason = r.get("stop_reason") or "none"
        counts[reason] = counts.get(reason, 0) + 1
    ordered = [k for k in STOP_REASON_ORDER if k in counts] + sorted(
        k for k in counts if k not in STOP_REASON_ORDER
    )
    return " · ".join(f"{STOP_REASON_LABELS.get(k, k)} {counts[k]}" for k in ordered)


def build_table(runs: list[dict]) -> str:
    """THE table: one row per level, one pass count per task, and the overall column."""
    tasks = task_names(runs)
    header = ["#", "level", "adds", *tasks, "overall", "avg steps", "avg tokens", "avg secs",
              "stop reasons"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for number, key, adds in level_rows(runs):
        mine = [r for r in runs if r["level"] == number]
        cells = [str(number), f"`{key}`", _md_cell(adds)]
        for task in tasks:
            of_task = [r for r in mine if r["task"] == task]
            cells.append(f"{sum(r['passed'] for r in of_task)}/{len(of_task)}")
        cells.append(f"**{_ratio(sum(r['passed'] for r in mine), len(mine))}**")
        cells.append(f"{_mean([r['steps'] for r in mine]):.1f}")
        cells.append(f"{_mean([r['total_tokens'] for r in mine]):,.0f}")
        cells.append(f"{_mean([r['seconds'] for r in mine]):.1f}")
        cells.append(stop_reasons(mine))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def infra_note(runs: list[dict]) -> str:
    """One paragraph, only when a cell was re-run after a daemon crash (else empty)."""
    retried = [r for r in runs if r.get("infra_retries")]
    if not retried:
        return ""
    cells = ", ".join(f"L{r['level']} {r['task']} r{r['run']}" for r in retried)
    return (
        f"\n{len(retried)} run(s) were re-run from scratch after an infrastructure error "
        f"(Ollama's model runner stopped; see `infra_errors` on the line): {cells}. Only "
        "the re-run counts; the crash is recorded, not scored.\n"
    )


def build_delta_table(runs: list[dict]) -> str:
    """"What the extra step bought": each level's overall pass rate against the row above.

    The row above is the next lower level that was RUN, and the cell names it
    ("vs L3: ..."), so a matrix run with --levels 0,3,4 shows that level 3 is being
    compared with level 0 rather than passing that off as one added thing. With the
    same number of runs on both rows the delta is in passes and in percentage points
    ("+2 passes (+22 pp)"). Two levels need not have the same number of runs (a level
    added later with --resume and a smaller --k, say), and then passes minus passes
    is not a quantity — 3/3 after 9/9 is not "-6 passes" — so the cell shows only the
    percentage points, with the two ratios beside them ("+0 pp (3/3 against 9/9)").
    The lowest level present is the baseline and has no delta.
    """
    header = ["#", "level", "adds", "overall", "vs the row above"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    previous: tuple[int, int, int] | None = None
    for number, key, adds in level_rows(runs):
        mine = [r for r in runs if r["level"] == number]
        passes, n = sum(r["passed"] for r in mine), len(mine)
        if previous is None:
            delta = "baseline"
        else:
            p_number, p_passes, p_n = previous
            pp = (100 * passes / n if n else 0.0) - (100 * p_passes / p_n if p_n else 0.0)
            if n == p_n:
                delta = f"vs L{p_number}: {passes - p_passes:+d} passes ({pp:+.0f} pp)"
            else:
                delta = f"vs L{p_number}: {pp:+.0f} pp ({passes}/{n} against {p_passes}/{p_n})"
        lines.append(
            f"| {number} | `{key}` | {_md_cell(adds)} | {_ratio(passes, n)} | {delta} |"
        )
        previous = (number, passes, n)
    return "\n".join(lines)


def levels_arg(numbers: list[int]) -> str:
    """[0,1,2,3,4,5] -> "0-5"; [0,2,5] -> "0,2,5": the --levels flag that names these."""
    if numbers and numbers == list(range(numbers[0], numbers[-1] + 1)) and len(numbers) > 1:
        return f"{numbers[0]}-{numbers[-1]}"
    return ",".join(str(n) for n in numbers)


def reproduce_command(runs: list[dict], k: int, out_arg: str) -> str:
    """The run_ladder.py command line that produces these rows again.

    Every flag is read off the rows, the step budget included: a matrix run with
    --max-steps 8 must not print a command that silently reproduces it with 12. The
    flag is added when the rows agree on one budget (they always do for one matrix;
    older lines without the field, or a directory that mixed budgets under --resume,
    get no flag rather than a wrong one).
    """
    numbers = [n for n, _, _ in level_rows(runs)]
    temperature = runs[0].get("temperature", DEFAULT_TEMPERATURE)
    seed_base = min(r["seed"] - r["run"] for r in runs)
    provider = runs[0].get("provider") or "ollama"
    model = runs[0].get("model") or DEFAULT_MODEL
    budgets = {r.get("max_steps") for r in runs if r.get("max_steps") is not None}
    cmd = (
        f"python benchmark/ladder/run_ladder.py --levels {levels_arg(numbers)} "
        f"--tasks {','.join(task_names(runs))} --k {k} --temperature {temperature} "
        f"--seed-base {seed_base} --provider {provider} --model {model}"
    )
    if provider == "openai":
        cmd += " --base-url <URL>"
    if len(budgets) == 1:
        cmd += f" --max-steps {budgets.pop()}"
    return cmd + f" --out {out_arg}"


def build_report(
    runs: list[dict],
    *,
    policy: dict | None = None,
    max_steps: int | None = None,
    k: int | None = None,
    out_arg: str = "<results dir>",
) -> str:
    """results.md: the fixed conditions, THE table, the delta table, how to reproduce.

    Everything in the header is read off the run lines themselves where possible
    (model, provider, temperature, seed base, the options actually sent), so a report
    rebuilt later with --report-only says what was run, not what the defaults are
    today. `policy` and `max_steps` are passed in by main (from levels.py and the
    flag) because the levels, not the runner, own them; `k` defaults to the largest
    run index seen plus one.
    """
    if not runs:
        raise ValueError("no runs to report")
    k = k if k is not None else max(r["run"] for r in runs) + 1
    models = sorted({str(r.get("model")) for r in runs})
    providers = sorted({str(r.get("provider")) for r in runs})
    temps = sorted({r.get("temperature", DEFAULT_TEMPERATURE) for r in runs})
    seed_base = min(r["seed"] - r["run"] for r in runs)
    steps = sorted({r.get("max_steps") for r in runs if r.get("max_steps") is not None})
    max_steps_text = (
        str(max_steps) if max_steps is not None
        else (", ".join(str(s) for s in steps) if steps else "see levels.py")
    )
    options = next((r["options"] for r in runs if r.get("options")), None)
    options_text = json.dumps(options, sort_keys=True) if options else "(not recorded)"
    policy_text = (
        json.dumps(policy, indent=2) if policy is not None
        else "see POLICY_RULES in benchmark/ladder/levels.py"
    )
    n_pass = sum(r["passed"] for r in runs)
    conditions = textwrap.fill(
        f"Model {', '.join(f'`{m}`' for m in models)} via {', '.join(providers)}; "
        f"temperature {', '.join(str(t) for t in temps)}; seed {seed_base} + run index "
        f"(run i of every level and task uses the same seed); k = {k} runs per level per "
        f"task; max_steps {max_steps_text}; generation options as sent on every request: "
        f"`{options_text}`. Grading is each task's own verify.py on the workspace copy "
        f"the level left behind; the task prompt, the fixture and the grader are the "
        f"same at every level. Headless: nobody answers a permission ASK, so ASK is DENY. "
        f"Every text the model reads at a level — tool descriptions and system prompt — "
        f"names only the tools that level has.",
        width=88,
    )
    return f"""# The capability ladder — results

Generated {datetime.now():%Y-%m-%d %H:%M} by `benchmark/ladder/run_ladder.py` from
{len(runs)} runs in `runs.jsonl` ({n_pass} passed). One JSON line per run; the tables are
rebuilt from the whole file each time (`--report-only`). Each run's full conversation is
in `transcripts/L<level>-<task>-r<run>.jsonl`, which is where to look when a number here
needs explaining.

## Fixed conditions

{conditions}

The permission policy, identical at every level (a safety property, not a capability):

```json
{policy_text}
```

## The ladder

Same model, same tasks, one more thing per level. The **overall** column is the number
that climbs.

{build_table(runs)}

Columns: `#`/level are the level's number and key (benchmark/ladder/levels.py); adds is
the ONE thing that level has and the level below does not; each task column is
passes/runs for that task; overall is passes over all of the level's runs; avg steps is
loop turns per run — one per model call inside the agent loop; the wrap-up call after an
unfinished run and a sub-agent's own calls are counted in tokens, not steps; a level-0/1
run is always one turn; avg tokens is prompt + output tokens per run, summed over every
model call, wrap-up and sub-agents included; avg secs is wall-clock per run, tool time
included; stop reasons is how the level's runs ended — `answered` (the model stopped on
its own and gave a final message, right or wrong: the loop's stop_reason `done` on the
run line), `max_steps` (the loop's budget ran out), `stuck` (the same call gave the same
result three times), `error` (the provider raised).

## What the extra step bought

Each row against the row above it — the next lower level that was run, named in the
cell — so the delta is the passes the added thing bought (or cost) when the two rows
were run the same number of times, and percentage points alone when they were not.
Read it with the transcripts: a delta of one run out of nine is one run.

{build_delta_table(runs)}
{infra_note(runs)}
## How to reproduce

```bash
cd <repo root>
source .venv/bin/activate          # the runner puts this interpreter's bin dir on PATH itself
{reproduce_command(runs, k, out_arg)}
```

Add `--resume` to continue a stopped matrix or to add a level to this directory;
`--report-only {out_arg}` rebuilds this file from `runs.jsonl` without running anything.
"""


# --- main -------------------------------------------------------------------------------


def default_out() -> Path:
    """benchmark/ladder/results/<YYYY-MM-DD_HHMM>/: a fresh directory per matrix."""
    return DEFAULT_RESULTS_ROOT / f"{datetime.now():%Y-%m-%d_%H%M}"


def _out_arg(out: Path) -> str:
    """The out directory as the reader would type it: relative to the repo when inside it."""
    try:
        return str(out.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(out)


def _policy_from_levels(levels_mod: ModuleType | None) -> dict | None:
    """levels.py's POLICY_RULES (the one policy every level uses), if it defines one."""
    if levels_mod is None:
        return None
    policy = getattr(levels_mod, "POLICY_RULES", None)
    return dict(policy) if isinstance(policy, dict) else None


def write_report(out: Path, runs: list[dict], **kwargs) -> Path:
    """Build results.md into `out` and return its path."""
    report = build_report(runs, out_arg=_out_arg(out), **kwargs)
    path = out / "results.md"
    path.write_text(report, encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the capability ladder: k runs of each coding task at each level, on one "
            "frozen model, graded by verify.py; writes runs.jsonl and results.md."
        )
    )
    parser.add_argument("--levels", default=DEFAULT_LEVELS, help="e.g. 0-5, 0,2,5 or 3 (default 0-5)")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS), help="comma-separated task names")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="runs per level per task")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE,
                        help="run i uses seed SEED_BASE + i at every level")
    parser.add_argument("--provider", choices=("ollama", "openai"), default="ollama")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama tag or served model name")
    parser.add_argument("--base-url", default=None, help="--provider openai: the server's /v1 root")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--out", type=Path, default=None,
                        help="results directory (default benchmark/ladder/results/<date_time>/)")
    parser.add_argument("--resume", action="store_true",
                        help="skip (level, task, run) triples already in <out>/runs.jsonl")
    parser.add_argument("--report-only", type=Path, default=None, metavar="DIR",
                        help="rebuild DIR/results.md from DIR/runs.jsonl and exit")
    parser.add_argument("--infra-retries", type=int, default=INFRA_RETRIES,
                        help="re-runs of a cell after a daemon crash (default 2; 0 = record the crash)")
    args = parser.parse_args(argv)

    if args.report_only is not None:
        out = args.report_only
        runs = read_runs(out / "runs.jsonl")
        if not runs:
            raise SystemExit(f"no runs in {out / 'runs.jsonl'}")
        try:
            levels_mod = load_levels()
        except Exception:  # noqa: BLE001 — the report must build even without levels.py
            levels_mod = None
        path = write_report(out, runs, policy=_policy_from_levels(levels_mod))
        print(build_table(runs))
        print(f"\nwrote {path}")
        return

    levels_mod = load_levels()
    levels = select_levels(levels_mod.LEVELS, parse_levels(args.levels))
    tasks = spike.discover_tasks([t.strip() for t in args.tasks.split(",") if t.strip()])
    # The step budget belongs to the levels (Level.run takes none), so the flag reaches
    # them through the module constant every level reads. If levels.py names it
    # differently the flag cannot act, and a reader must be told rather than misled.
    if hasattr(levels_mod, "MAX_STEPS"):
        levels_mod.MAX_STEPS = args.max_steps
    elif args.max_steps != DEFAULT_MAX_STEPS:
        print(
            f"warning: levels.py has no MAX_STEPS; --max-steps {args.max_steps} has no effect",
            file=sys.stderr, flush=True,
        )

    out: Path = args.out if args.out is not None else default_out()
    if (out / "runs.jsonl").exists() and not args.resume:
        raise SystemExit(
            f"{out / 'runs.jsonl'} exists; add --resume to continue it or pick a fresh --out"
        )
    factory = make_provider_factory(args.provider, args.model, args.temperature, args.base_url)

    total = len(levels) * len(tasks) * args.k
    print(
        f"ladder: {len(levels)} level(s) x {len(tasks)} task(s) x k={args.k} = {total} runs "
        f"on {args.model} via {args.provider} -> {out}",
        flush=True,
    )
    failure = warm_up(factory(args.seed_base))
    if failure:
        print(f"warm-up failed: {failure}", file=sys.stderr, flush=True)

    runs = run_matrix(
        levels, tasks, args.k, factory, out,
        seed_base=args.seed_base,
        temperature=args.temperature,
        max_steps=args.max_steps,
        model=args.model,
        provider_label=args.provider,
        resume=args.resume,
        infra_retries=args.infra_retries,
    )
    path = write_report(
        out, runs, policy=_policy_from_levels(levels_mod), max_steps=args.max_steps, k=args.k
    )
    print()
    print(build_table(runs))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
