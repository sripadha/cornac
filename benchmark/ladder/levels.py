"""The six levels of the capability ladder — the single source of truth for each one.

The experiment, in one paragraph
--------------------------------
Take one broken program and one small model (qwen3.5:4b, frozen with round three's
sampling settings). Ask the model to fix it six times, and each time give it exactly
one more kind of help than the time before. Grade every attempt the same way — the
task's own verify.py, which runs the tests and checks the fix is general — and count
how often it succeeds at each level. Because each level adds ONE thing, a jump in the
pass rate between two neighbouring levels can only have come from that one thing.
That table of six numbers is the point of the whole project: the same weak model,
climbing. Round three (benchmark/spike/results_round3, 9/9 on these three tasks with
the Week 4b harness on) is roughly what level 4 should reproduce; the 12/27 -> 26/27
uplift in docs/observations.md is what the ladder decomposes into its parts.

The six levels
--------------
    #  key       adds (exactly one thing)
    0  bare      nothing: the question and the code pasted in, one model call
    1  prompted  a system prompt (persona + "Workspace: <files>")
    2  one_tool  one tool, run_bash: the files are no longer pasted in; the model
                 can ACT, with a shell and nothing else (the agent loop begins here)
    3  tools     the full read/write tool set of the round-two harness, guards off
    4  harness   better tools (syntax gate, change report, edit_file) + loop guards
                 (nudge, repeat note, repeat hard stop, wrap-up, context clearing)
    5  agents    sub-agents: spawn_agent, and one sentence saying it exists

The `adds` text on each Level is the table's third column and is written for a
reader outside this repo (the table is the LinkedIn post); the round and week
references that place each level in the project's history stay here, in the
docstrings.

Levels 0 and 1 are "one-shot": a single provider.complete() call with no tools. The
model cannot look at anything, so the files are pasted into the prompt as fenced
blocks, and it is asked to reply with corrected files in the same format;
apply_answer() then writes those blocks into the workspace so verify.py has
something to grade. Levels 2 to 5 are real Agent runs (cornac.core.agent) and the
model changes files itself.

What stays constant, at every level: the task prompt (from task.md, untouched), a
fresh copy of the fixture per run (the runner's job), grading by verify.py AFTER
run() returns (the runner's job too — a level never grades itself), max_steps 12,
no approver (headless: an ASK from the policy is a DENY), and one permission policy,
POLICY_RULES. The policy is a safety property, not a capability — the shell
blocklist stops `rm -rf` at level 2 exactly as at level 5 — so it does not count as
one of the "things" a level adds.

One rule about words: every text the model reads names only tools that level has.
A tool description that says "to change a file use edit_file" at a level without
edit_file, or a system prompt that says "run_python already runs in the workspace"
at a level without run_python, makes a model call a tool that does not exist
("Unknown tool" from the registry: a burnt step and a tool error) or believe it
cannot do what it can — a penalty charged to the wrong row. So make_tool builds
levels 2 and 3 with round two's tool descriptions (no edit_file/write_file hint) and
levels 4-5 with today's, write_file's description follows its flags (files.py), and
Workspace.describe() is given the level's tool names. A test walks every agent level
and checks that no description and no system prompt names an absent tool.

Why the level is a data object and not six scripts
-------------------------------------------------
Six example scripts (examples/stage_*.py), the matrix runner
(benchmark/ladder/run_ladder.py) and the tests all have to agree on what "level 3"
means, down to the exact persona text and which write_file flags are on. Written
six times, those details drift; a `Level` written once and imported everywhere
cannot. Level.run(provider, workspace, task_prompt, transcript_path) builds the
model call or the agent for that level and returns a LevelOutcome — the numbers
the table needs, nothing graded — and the runner does the same thing with every
level, which is what makes the comparison fair.

Two ways to import this file. As `benchmark.ladder.levels` (the repo root is on
sys.path: the dev venv's editable install puts it there, and the runner and the
examples bootstrap it themselves — this module adds nothing to sys.path), or by
file path with importlib the way the runner loads run_spike.py; in that case
register the module in sys.modules BEFORE executing it, because the @dataclasses
here resolve their `from __future__ import annotations` strings through
sys.modules[cls.__module__].
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from cornac import Agent, HookBus, Message, Policy, ToolRegistry, Usage
from cornac.providers.base import Provider
from cornac.tools.base import Tool
from cornac.tools.builtin.files import EditFile, Grep, ListDir, ReadFile, WriteFile
from cornac.tools.builtin.python_exec import RunPython
from cornac.tools.builtin.shell import RunBash
from cornac.tools.builtin.spawn import SpawnAgent
from cornac.tools.workspace import Workspace, WorkspaceError
from cornac.transcript import TranscriptWriter

# --- the fixed conditions -----------------------------------------------------------

# The step budget of every agent level: round three's (run_spike.MAX_STEPS). Twelve
# is enough for run tests -> read -> edit -> run tests with room to recover from a
# mistake or two, and small enough that a model going in circles is stopped soon.
MAX_STEPS = 12

# One policy for every level. Headless, so every tool a task can need is a plain
# allow (nobody is there to answer an ASK, and the loop turns an unanswered ASK into
# a DENY); default deny, so a tool nobody thought about — the web tools — cannot run.
# run_bash keeps the destructive-command blocklist: the model runs real shell commands
# in a real temp directory. Identical at every level, on purpose (module docstring).
POLICY_RULES: dict = {
    "default": "deny",
    "tools": {
        "read_file": "allow",
        "write_file": "allow",
        "edit_file": "allow",
        "list_dir": "allow",
        "grep": "allow",
        "run_python": "allow",
        "spawn_agent": "allow",
        "run_bash": {
            "deny": ["rm -rf", "sudo", "mkfs", "dd if=", ":(){"],
            "default": "allow",
        },
    },
}

# --- the words the model is given ---------------------------------------------------

# Levels 0 and 1. The model cannot read the disk, so the files go INTO the message,
# each behind a `# file:` line; the same line is what the reply must carry, so that
# apply_answer can tell which block is which file.
FILES_HEADER = "The files in the workspace:"
ANSWER_FORMAT = (
    "Reply with the complete corrected version of every file you change, each as a "
    "fenced code block whose first line is `# file: <path>`. Do not include files you "
    "did not change."
)
# The persona, in one core shared by every prompted level plus a tail per kind of
# level. The core carries the brief — who the model is, what to fix, how little to
# change, what never to touch — and is the SAME words at level 1 and at level 2, so
# the 1 -> 2 row measures the tool and not a reworded brief (an early draft dropped
# "changing as little as possible" exactly where the model first gained the means to
# rewrite whole files, which is the round-two failure mode). Only the tail changes,
# and only because it must: a model that can only answer is told the answer format;
# a model with tools is told the procedure and what to report.
PERSONA_CORE = (
    "You are a careful software engineer. Fix the failing test by changing as little as "
    "possible in the source, never the test file."
)
# Level 1's one addition over level 0: the persona, for a model that can only answer.
ONE_SHOT_TAIL = "Reply only with the corrected file(s) in the fenced-block format requested."
ONE_SHOT_PERSONA = PERSONA_CORE + " " + ONE_SHOT_TAIL

# Levels 2 to 5, identical text, followed by Workspace.describe() (the absolute root,
# so the model does not guess a cwd — round two had one run `cd /workspace`). Only
# level 5 adds a sentence, DELEGATION_SENTENCE, and nothing else.
TOOLS_TAIL = (
    "You are working in a sandboxed workspace: run the tests, find the bug, fix it, run "
    "the tests again, then say which function you fixed and what was wrong."
)
TOOLS_PERSONA = PERSONA_CORE + " " + TOOLS_TAIL
DELEGATION_SENTENCE = (
    "You may hand a self-contained sub-task (searching, reading, running the tests) to "
    "a sub-agent with spawn_agent; you keep the edits."
)

# What is never pasted into a prompt and never counted as a changed file: pytest's
# caches and Python's bytecode. They appear the moment the tests run, and a level
# that listed them would look as if it had changed files it never touched.
SKIP_DIRS = frozenset({"__pycache__", ".pytest_cache"})
SKIP_SUFFIXES = frozenset({".pyc"})

# The event name of the one transcript record a one-shot level writes.
ONE_SHOT_EVENT = "one_shot"
# The record written before a provider failure is re-raised (both kinds of level).
RUN_ERROR_EVENT = "run_error"


# --- the loop guards ----------------------------------------------------------------


@dataclass(frozen=True)
class LoopSettings:
    """The five Agent knobs the ladder turns on at level 4.

    Each is a Week 4b/4c behaviour of the loop (cornac/core/agent.py), and each has
    an off position so the levels below can measure the model WITHOUT it: the
    continue nudge (max_nudges), the "you already made this exact call" note
    (repeat_note), the hard stop after identical calls (max_repeats, 0 = never), the
    post-mortem model call of an unfinished run (wrap_up), and context clearing
    (context_window: 0 = off even when the provider knows its window, None = use the
    provider's — Ollama's num_ctx, 8192 in the frozen settings).
    """

    max_nudges: int
    repeat_note: bool
    max_repeats: int
    wrap_up: bool
    context_window: int | None


# Levels 2 and 3: the loop as a bare while-loop. Levels 4 and 5: the loop as shipped.
GUARDS_OFF = LoopSettings(
    max_nudges=0, repeat_note=False, max_repeats=0, wrap_up=False, context_window=0
)
GUARDS_ON = LoopSettings(
    max_nudges=1, repeat_note=True, max_repeats=3, wrap_up=True, context_window=None
)


# --- what a level hands back --------------------------------------------------------


@dataclass
class LevelOutcome:
    """What one run of one level produced — the numbers for the table, NOT a grade.

    The runner grades the workspace with verify.py after run() returns, so nothing
    here says whether the fix was right. `passed_precheck` says whether there was
    anything to grade at all: False when a one-shot level's reply carried no file
    block that could be written (the model never answered in the format, so the
    workspace is untouched), True otherwise. The count fields mirror RunResult
    (cornac/core/result.py) so a one-shot level and an agent level fill the same
    columns: a one-shot run is 1 step, 0 tool calls, 0 nudges, 0 children.

    `files_applied` is the list of files whose bytes changed on disk during the run
    — created, modified or deleted, caches excluded — whichever way they changed:
    apply_answer at levels 0-1, write_file/edit_file or a shell redirect at 2-5.
    One meaning at every level, measured the same way (a snapshot before and after),
    because a run_bash edit leaves no other trace. `rejected` counts the blocks of a
    one-shot reply whose path pointed outside the workspace (always 0 for an agent
    level, where the Workspace fence refuses such a path inside the tool instead).
    `messages` is the whole conversation, for a reader who wants more than numbers.
    """

    passed_precheck: bool
    final_text: str | None
    stop_reason: str
    steps: int
    usage: Usage
    duration: float
    tool_calls: int = 0
    tool_errors: int = 0
    nudges: int = 0
    children: int = 0
    files_applied: list[str] = field(default_factory=list)
    rejected: int = 0
    digest: str | None = None
    summary: str | None = None
    messages: list[Message] = field(default_factory=list)


# --- the files of a workspace: listing, pasting, and what changed --------------------


def _iter_files(workspace: Workspace):
    """(relative posix path, real path) for every file under the root, sorted, minus
    caches, bytecode, and anything a symlink points at outside the root."""
    for path in sorted(workspace.root.rglob("*")):
        rel = path.relative_to(workspace.root)
        if SKIP_DIRS.intersection(rel.parts[:-1]) or path.suffix in SKIP_SUFFIXES:
            continue
        try:
            real = workspace.resolve(path)  # a link that leaves the sandbox is not its file
        except WorkspaceError:
            continue
        if not real.is_file():
            continue
        yield rel.as_posix(), real


def text_files(workspace: Workspace) -> list[tuple[str, str]]:
    """(relative path, contents) of every UTF-8 text file in the workspace, sorted.

    Text only, because the files are going into a prompt: a model cannot read bytes
    it is shown as mojibake, and a fixture may carry a binary asset. A file that is
    not UTF-8 is left out rather than pasted with replacement characters.
    """
    out: list[tuple[str, str]] = []
    for rel, real in _iter_files(workspace):
        try:
            out.append((rel, real.read_bytes().decode("utf-8")))
        except (UnicodeDecodeError, OSError):
            continue
    return out


def workspace_files(workspace: Workspace) -> list[str]:
    """The relative paths text_files() would paste — level 1's "Workspace:" line."""
    return [rel for rel, _ in text_files(workspace)]


def fenced_file(rel: str, text: str) -> str:
    """One file as the fenced block the one-shot prompt uses and the reply must echo.

    The fence is one backtick longer than the longest run of backticks inside the
    file, so a file that itself contains ``` cannot close the block early. The
    language tag is cosmetic — models write ```python back, and apply_answer accepts
    any tag or none. The first line inside the fence is the `# file:` marker the
    reply is asked to reproduce.
    """
    longest = max((len(run) for run in re.findall(r"`{3,}", text)), default=2)
    fence = "`" * (longest + 1)
    lang = "python" if rel.lower().endswith(".py") else ""
    body = text if text.endswith("\n") else text + "\n"
    return f"{fence}{lang}\n# file: {rel}\n{body}{fence}"


def bare_user_message(task_prompt: str, workspace: Workspace) -> str:
    """The one user message of levels 0 and 1: prompt, every file, the answer format.

    Level 1 adds a system prompt and NOTHING else, so this text is shared by both
    levels and a test asserts they send it byte for byte the same. The task prompt
    comes first, untouched from task.md, because it is the same prompt every other
    level gets; the files follow because the model cannot look at them; the format
    request comes last because it is the instruction most likely to be obeyed there.
    """
    blocks = "\n\n".join(fenced_file(rel, text) for rel, text in text_files(workspace))
    return f"{task_prompt}\n\n{FILES_HEADER}\n\n{blocks}\n\n{ANSWER_FORMAT}"


def snapshot(workspace: Workspace) -> dict[str, str]:
    """Relative path -> SHA-1 of its bytes for every file that counts (caches excluded)."""
    out: dict[str, str] = {}
    for rel, real in _iter_files(workspace):
        try:
            out[rel] = hashlib.sha1(real.read_bytes()).hexdigest()
        except OSError:
            continue
    return out


def changed_files(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """The paths whose hash differs between two snapshots: created, modified or deleted."""
    return sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))


# --- turning a one-shot reply into files ---------------------------------------------

# A fenced block: an opening fence of three or more backticks at the start of a line
# (any info string after it), the body, and a closing fence of the SAME length alone
# on its line. The backreference is what lets a ````-fenced file contain ``` lines.
_FENCE = re.compile(
    r"^(?P<fence>`{3,})[^\n]*\n(?P<body>.*?)\n(?P=fence)[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
# The marker line: `# file: <path>`, spaces optional, the path possibly quoted, and
# `file` in any case — a model echoing a heading writes `# File:` as readily as
# `# file:`, and the marker is the harness's answer protocol, not the task, so its
# spelling must not decide a row. The path itself keeps its case.
_FILE_LINE = re.compile(r"^\s*#\s*file:\s*(?P<path>\S.*?)\s*$", re.IGNORECASE)


@dataclass
class AppliedAnswer:
    """What apply_answer did: the files it wrote (workspace-relative, each once) and
    the paths it refused because they pointed outside the workspace."""

    written: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)


# Any fence line at all, wherever it is: whether the reply spoke the fenced protocol.
# file_blocks falls back to the bare `# file:` markers only when there is none.
_FENCE_LINE = re.compile(r"^`{3,}", re.MULTILINE)


def file_blocks(text: str) -> list[tuple[str, str]]:
    """(path, contents) for every file block of `text`: a fenced block marked `# file: ...`.

    Line endings are normalised to "\\n" first: a model (or a Windows terminal) may
    emit CRLF, and a regex that expected "\\n" before the closing fence would then
    find no blocks at all. The marker is the first NON-BLANK line inside the fence
    (a blank line after the opening fence is a formatting habit, not a different
    answer), or else the last non-blank line right BEFORE the opening fence, where a
    model that took the marker for a heading puts it; either way it matches in any
    case (_FILE_LINE). A block with no marker in either place — a model showing a
    snippet, or its shell command — is not a file and is skipped. The contents get a
    trailing newline back (the regex consumed the one before the closing fence), so
    a file round-trips through fenced_file and back byte for byte.

    A reply with NO fence line anywhere is read by its markers alone: each `# file:`
    line opens a file that runs to the next marker line or the end of the text
    (_unfenced_blocks). The first smoke run of the ladder (2026-09-27) had the bare
    model at level 0 answer with the complete, correct calc.py under a
    `# file: calc.py` line and no fence at all, and the row failed on the missing
    decoration rather than on the code. The marker and the fence are the harness's
    answer protocol, not the task: the deliverable is the file, and verify.py grades
    the workspace, so a boundary this fallback guesses wrong can only fail a row,
    never pass one. The fallback stays off the moment the reply uses any fence — a
    fence that never closes (a reply cut by the output cap), a fenced snippet — since
    a model speaking the protocol is held to it, and then only fenced blocks count.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks: list[tuple[str, str]] = []
    for match in _FENCE.finditer(text):
        lines = match.group("body").split("\n")
        first = next((i for i, line in enumerate(lines) if line.strip()), None)
        if first is None:
            continue
        marker = _FILE_LINE.match(lines[first])
        if marker is not None:
            body = lines[first + 1:]
        else:
            before = text[: match.start()].split("\n")
            last = next((line for line in reversed(before) if line.strip()), "")
            marker = _FILE_LINE.match(last)
            if marker is None:
                continue
            body = lines
        blocks.append((_marker_path(marker), _joined(body)))
    if not _FENCE_LINE.search(text):
        blocks = _unfenced_blocks(text)
    return blocks


def _marker_path(marker: re.Match) -> str:
    """The path a `# file:` marker names, without the quotes a model may put round it."""
    return marker.group("path").strip("`'\"")


def _joined(lines: list[str]) -> str:
    """Lines back to file contents, with the one trailing newline the fence consumed."""
    rest = "\n".join(lines)
    return rest + "\n" if rest else ""


def _unfenced_blocks(text: str) -> list[tuple[str, str]]:
    """The marker-delimited files of a reply that used no fence at all (see file_blocks).

    Trailing blank lines are dropped before the newline goes back: a fenced block
    ends exactly at its fence, an unfenced one wherever the model stopped typing.
    """
    lines = text.split("\n")
    starts = [i for i, line in enumerate(lines) if _FILE_LINE.match(line)]
    blocks: list[tuple[str, str]] = []
    for n, start in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        body = lines[start + 1:end]
        while body and not body[-1].strip():
            body.pop()
        blocks.append((_marker_path(_FILE_LINE.match(lines[start])), _joined(body)))
    return blocks


def apply_answer(text: str, workspace: Workspace) -> AppliedAnswer:
    """Write every `# file:` block of a one-shot reply into the workspace.

    This is the hands a level-0 or level-1 model does not have: it can only say
    what the file should be, and something has to put those bytes on disk before
    verify.py can run the tests against them. Every path goes through
    Workspace.resolve(), the same fence every tool uses, so a block for
    `../../.bashrc` or `/etc/passwd` is refused and counted rather than written —
    the model's text is untrusted output at level 0 exactly as a tool call is at
    level 3. Nothing is checked beyond that: a block that overwrites the test file
    is written, and verify.py's AST check is what catches it, because "did the model
    edit the tests?" is part of what the level measures. Text with no blocks writes
    nothing.
    """
    applied = AppliedAnswer()
    for path, content in file_blocks(text):
        try:
            target = workspace.resolve(path)
        except WorkspaceError:
            applied.rejected.append(path)
            continue
        if target == workspace.root or target.is_dir():
            applied.rejected.append(path)  # "# file: ." is not a file
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8", newline="")
        except OSError:
            applied.rejected.append(path)
            continue
        rel = workspace.relative(target)
        if rel not in applied.written:
            applied.written.append(rel)
    return applied


# --- the tools, by name -------------------------------------------------------------


def make_tool(name: str, workspace: Workspace, harness: bool) -> Tool:
    """One built-in tool for a level's registry.

    `harness` is level 4's one addition on the tool side: with it, write_file
    compiles a .py file before writing and reports what a rewrite removed, and
    edit_file compiles too. Without it (level 3) the same tools behave as the
    round-two harness's did — a broken rewrite lands on disk and says "wrote N
    chars" — which is what round two measured and what level 3 must reproduce.
    The flags exist on the tools for exactly this comparison (see files.py).

    The same flag chooses what run_bash and run_python SAY about themselves. Their
    shipped descriptions are Week 4b text ("does NOT edit source files: use edit_file
    or write_file"), written for a registry that has both. Level 2 has neither —
    the shell is its only editor — and level 3 has no edit_file; round two's
    descriptions (git c381aed) named no tool at all. So `edit_hint=harness`: levels
    2 and 3 read round two's text verbatim, levels 4 and 5 today's, and no level is
    pointed at a tool it does not have (module docstring, "One rule about words").
    """
    if name == "read_file":
        return ReadFile(workspace)
    if name == "write_file":
        return WriteFile(workspace, syntax_gate=harness, change_report=harness)
    if name == "edit_file":
        return EditFile(workspace, syntax_gate=harness)
    if name == "list_dir":
        return ListDir(workspace)
    if name == "grep":
        return Grep(workspace)
    if name == "run_bash":
        return RunBash(workspace, edit_hint=harness)
    if name == "run_python":
        return RunPython(workspace, edit_hint=harness)
    if name == "spawn_agent":
        # Binds to its agent in Agent.__init__. A child inherits every loop knob of
        # LoopSettings from its parent (spawn.INHERITED_SETTINGS + max_nudges), so
        # level 5's children run with level 4's guards, not the Agent's defaults.
        return SpawnAgent()
    raise ValueError(f"no such ladder tool: {name!r}")


class _ToolCounter:
    """A post_tool_use observer: how many tool calls a run made and how many failed.

    Counted from the hook rather than from the message list so that a sub-agent's
    calls count too — spawn_agent forwards post_tool_use to the parent's bus, but
    a child's messages never enter the parent's list. A call failed when the loop
    flagged it (a denial, a veto, a raise) or when the tool itself said so with an
    "Error:" first line — the same two kinds the spike's tool-error rate counts. A
    non-zero exit code is NOT a failure here: the first pytest run of every good
    fix exits 1, and that is the model doing its job.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.errors = 0

    def __call__(self, call, result) -> None:
        self.calls += 1
        if result.is_error or result.content.lstrip().startswith("Error:"):
            self.errors += 1


# --- a level ------------------------------------------------------------------------


@dataclass(frozen=True)
class Level:
    """One rung: what it adds, and how to run the frozen model at that level.

    The four description fields (number, key, title, adds) are what the table and
    the examples print. The four switches below them are the whole definition of
    the level's behaviour: `tools` names the registry in order (empty = one-shot,
    no agent), `prompted` says whether a system prompt is sent (False at level 0
    only), `harness` turns on the gated tools and the loop guards (levels 4-5), and
    `delegation` adds the one sentence about spawn_agent (level 5). Everything a
    level does is derived from these, so two levels that differ in one switch
    differ in one thing.
    """

    number: int
    key: str
    title: str
    adds: str
    tools: tuple[str, ...] = ()
    prompted: bool = True
    harness: bool = False
    delegation: bool = False

    # --- what the level is ------------------------------------------------------------

    @property
    def one_shot(self) -> bool:
        """True for levels 0 and 1: one model call, no tools, apply_answer for hands."""
        return not self.tools

    @property
    def loop(self) -> LoopSettings | None:
        """The loop guards this level runs with; None for a one-shot level."""
        if self.one_shot:
            return None
        return GUARDS_ON if self.harness else GUARDS_OFF

    # --- the words -----------------------------------------------------------------------

    def system_prompt_for(self, workspace: Workspace) -> str | None:
        """The system prompt for a run in `workspace`, or None at level 0.

        Per workspace, not a constant, because Workspace.describe() names the
        absolute root of THIS run's temp directory and level 1's line lists THIS
        run's files. The fixed text around them is identical for every run.
        """
        if not self.prompted:
            return None
        if self.one_shot:
            return f"{ONE_SHOT_PERSONA}\n\nWorkspace: {', '.join(workspace_files(workspace))}"
        persona = TOOLS_PERSONA + (" " + DELEGATION_SENTENCE if self.delegation else "")
        # describe() is told which tools exist so its cwd sentence names only those:
        # at level 2 "run_bash already runs inside that directory", from level 3 up
        # the shipped "run_bash and run_python ..." (module docstring, "One rule
        # about words"). The persona is the same words at every agent level.
        return persona + "\n\n" + workspace.describe(tools=self.tools)

    def user_message(self, task_prompt: str, workspace: Workspace) -> str:
        """What the model is asked: the task prompt alone when it has tools to look
        with, the prompt plus every file when it does not (levels 0 and 1)."""
        if self.one_shot:
            return bare_user_message(task_prompt, workspace)
        return task_prompt

    # --- the agent (levels 2-5) ----------------------------------------------------------

    def registry(self, workspace: Workspace) -> ToolRegistry:
        """The level's tools, built for `workspace`, in the table's order."""
        return ToolRegistry([make_tool(name, workspace, self.harness) for name in self.tools])

    def build_agent(self, provider: Provider, workspace: Workspace, hooks: HookBus | None = None) -> Agent:
        """The Agent a run of this level uses — the one place its settings are chosen.

        max_steps, the policy and the missing approver (headless) are the fixed
        conditions; the registry and the five loop knobs are the level's. Every knob
        is passed explicitly, never left to the Agent's default, so a default that
        changes later cannot silently change what a level measures.
        """
        if self.one_shot:
            raise ValueError(f"level {self.number} ({self.key}) is one model call and builds no agent")
        loop = self.loop
        assert loop is not None
        return Agent(
            provider=provider,
            registry=self.registry(workspace),
            system_prompt=self.system_prompt_for(workspace),
            max_steps=MAX_STEPS,
            policy=Policy(POLICY_RULES),
            hooks=hooks,
            max_nudges=loop.max_nudges,
            repeat_note=loop.repeat_note,
            max_repeats=loop.max_repeats,
            wrap_up=loop.wrap_up,
            context_window=loop.context_window,
        )

    # --- one run -------------------------------------------------------------------------

    def run(
        self,
        provider: Provider,
        workspace: Workspace,
        task_prompt: str,
        transcript_path: str | Path,
        *,
        hooks: HookBus | None = None,
    ) -> LevelOutcome:
        """Run the model once at this level, in `workspace`, and account for it.

        The workspace must be a fresh copy of the task's fixture (the runner makes
        one per run) — the run changes files in it, and the caller grades those
        files afterwards. A JSONL transcript goes to `transcript_path`: every hook
        event for an agent level (cornac.transcript), one "one_shot" record for
        levels 0-1. A provider failure is written to the transcript as a
        "run_error" record and then re-raised: the runner records it as a failed
        run, the way run_spike does, and nothing here pretends it was an answer.

        `hooks` (keyword only; the contract's four positional arguments are
        unchanged) lets a caller watch an agent level's run AS IT HAPPENS: a HookBus
        with the caller's handlers already registered (post_tool_use to print each
        tool call, on_assistant_message for each model turn, ...) becomes the run's
        bus, and the level adds its own observers — the tool counter and the
        transcript writer — next to them. A twelve-step run on a 4B model takes
        minutes, and the examples (examples/stage_2..5) print the calls live rather
        than replaying the transcript afterwards. Use a fresh bus per run: the
        level's observers stay registered on it. Levels 0-1 make no agent and fire
        no hooks, so `hooks` is ignored there (the reply is the whole run).
        """
        path = Path(transcript_path)
        if self.one_shot:
            return _run_one_shot(self, provider, workspace, task_prompt, path)
        return _run_agent(self, provider, workspace, task_prompt, path, hooks=hooks)


def _run_one_shot(
    level: Level, provider: Provider, workspace: Workspace, task_prompt: str, transcript_path: Path
) -> LevelOutcome:
    """Levels 0 and 1: one model call, then apply_answer for the model's hands."""
    before = snapshot(workspace)
    messages: list[Message] = []
    system = level.system_prompt_for(workspace)
    if system is not None:
        messages.append(Message.system(system))
    user = Message.user(level.user_message(task_prompt, workspace))
    messages.append(user)

    writer = TranscriptWriter(transcript_path)
    started = time.perf_counter()
    try:
        reply = provider.complete(messages, [])  # no tools: the model can only answer
    except Exception as exc:
        writer.record(RUN_ERROR_EVENT, level=level.number, error=f"{type(exc).__name__}: {exc}")
        writer.close()
        raise
    duration = time.perf_counter() - started
    if reply.tool_calls:
        # Offered no tools and asked for one anyway: nothing runs, and the stored
        # reply drops the request so the message list stays a valid conversation.
        reply = replace(reply, tool_calls=[])
    messages.append(reply)

    applied = apply_answer(reply.text or "", workspace)
    files = changed_files(before, snapshot(workspace))
    writer.record(
        ONE_SHOT_EVENT,
        level=level.number,
        key=level.key,
        provider=provider.name,
        model=getattr(provider, "model", None),
        system_prompt=system,
        user_message=user.text,
        reply=reply,
        written=applied.written,
        rejected=applied.rejected,
        files_applied=files,
        duration=round(duration, 3),
    )
    writer.close()
    return LevelOutcome(
        passed_precheck=bool(applied.written),
        final_text=reply.text,
        stop_reason="done",
        steps=1,
        usage=reply.usage or Usage(),
        duration=duration,
        files_applied=files,
        rejected=len(applied.rejected),
        messages=messages,
    )


def _run_agent(
    level: Level,
    provider: Provider,
    workspace: Workspace,
    task_prompt: str,
    transcript_path: Path,
    hooks: HookBus | None = None,
) -> LevelOutcome:
    """Levels 2 to 5: an Agent run, observed by a tool counter and a transcript writer
    (and by whatever the caller registered on `hooks`, which becomes the run's bus)."""
    before = snapshot(workspace)
    counter = _ToolCounter()
    bus = hooks if hooks is not None else HookBus()
    bus.on("post_tool_use", counter)
    agent = level.build_agent(provider, workspace, hooks=bus)
    writer = TranscriptWriter(transcript_path).attach(agent)  # run_config first, then every event
    try:
        result = agent.run(task_prompt)
    except Exception as exc:
        writer.record(RUN_ERROR_EVENT, level=level.number, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        writer.close()
    return LevelOutcome(
        passed_precheck=True,
        final_text=result.text,
        stop_reason=result.stop_reason,
        steps=result.steps,
        usage=result.usage,
        duration=result.duration,
        tool_calls=counter.calls,
        tool_errors=counter.errors,
        nudges=result.nudges,
        children=result.children,
        files_applied=changed_files(before, snapshot(workspace)),
        digest=result.digest,
        summary=result.summary,
        messages=list(agent.messages),
    )


# --- the ladder ---------------------------------------------------------------------

# The round-two tool set (level 3) and the round-three one (level 4), in the order the
# model sees them. edit_file is the one tool level 4 adds; spawn_agent is level 5's.
ROUND_TWO_TOOLS = ("read_file", "write_file", "list_dir", "grep", "run_bash", "run_python")
HARNESS_TOOLS = ("read_file", "write_file", "edit_file", "list_dir", "grep", "run_bash", "run_python")

# Indexed by level number: LEVELS[3] is level 3. The `adds` text is the table's
# third column and is worded as the one thing that level has that the one below does
# not — for a reader who has never seen this repo (no round or week numbers; those are
# in the docstrings), and short enough for a table cell (ADDS_MAX_CHARS, the runner's
# cell cap, so it is never cut with an ellipsis in results.md).
ADDS_MAX_CHARS = 90
LEVELS: tuple[Level, ...] = (
    Level(
        number=0,
        key="bare",
        title="Bare model",
        adds="nothing: the question and the code, one model call",
        prompted=False,
    ),
    Level(
        number=1,
        key="prompted",
        title="System prompt",
        adds="a system prompt",
    ),
    Level(
        number=2,
        key="one_tool",
        title="One tool",
        adds="one tool, run_bash: the files are no longer pasted in; the agent loop begins here",
        tools=("run_bash",),
    ),
    Level(
        number=3,
        key="tools",
        title="Full tool set",
        adds="the file tools: read_file, write_file, list_dir, grep, run_python (no syntax check)",
        tools=ROUND_TWO_TOOLS,
    ),
    Level(
        number=4,
        key="harness",
        title="Single-agent harness",
        adds="edit_file, syntax gate, change report, nudge, repeat note/stop, wrap-up, context clearing",
        tools=HARNESS_TOOLS,
        harness=True,
    ),
    Level(
        number=5,
        key="agents",
        title="Sub-agents",
        adds="sub-agents",
        tools=HARNESS_TOOLS + ("spawn_agent",),
        harness=True,
        delegation=True,
    ),
)


def get_level(which: int | str) -> Level:
    """A level by number (3) or key ("tools"); a KeyError names what exists."""
    for level in LEVELS:
        if which == level.number or which == level.key:
            return level
    raise KeyError(
        f"no level {which!r}; levels are "
        + ", ".join(f"{lv.number} ({lv.key})" for lv in LEVELS)
    )


__all__ = [
    "ADDS_MAX_CHARS",
    "ANSWER_FORMAT",
    "DELEGATION_SENTENCE",
    "FILES_HEADER",
    "GUARDS_OFF",
    "GUARDS_ON",
    "HARNESS_TOOLS",
    "LEVELS",
    "MAX_STEPS",
    "ONE_SHOT_EVENT",
    "ONE_SHOT_PERSONA",
    "ONE_SHOT_TAIL",
    "PERSONA_CORE",
    "POLICY_RULES",
    "ROUND_TWO_TOOLS",
    "RUN_ERROR_EVENT",
    "TOOLS_PERSONA",
    "TOOLS_TAIL",
    "AppliedAnswer",
    "Level",
    "LevelOutcome",
    "LoopSettings",
    "apply_answer",
    "bare_user_message",
    "changed_files",
    "fenced_file",
    "file_blocks",
    "get_level",
    "make_tool",
    "snapshot",
    "text_files",
    "workspace_files",
]
