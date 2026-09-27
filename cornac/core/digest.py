"""The digest — a model-free account of an unfinished run (Week 4c).

Why this exists
---------------
Until Week 4c a run that hit max_steps handed back text=None and nothing else. That
was the honest choice (the loop refuses to invent an answer), but it threw away the
one thing the run did produce: a transcript of everything that was tried. A parent
whose sub-agent ran out of steps learned "it gave up after 10 steps" and had to guess
what those steps were, or start a fresh child from zero; a human at the terminal saw
"did not finish" and had to scroll back through raw messages. The 2026 harness
comparison flagged this as the one place where OpenCode and Kilo, which hand back a
summary of the partial work, beat cornac.

There are two ways to write such an account. The model can be asked to — and it is,
see WRAP_UP_MESSAGE in core/agent.py — because it knows what it was trying to do. But
a model that has just run out of steps is often a model that was going in circles,
and a model in that state will happily write a summary that sounds like success. So
the harness writes its own account FIRST, from the transcript alone, with no model
call: which tools were called with what, which ones errored, what the last error
said, which files were changed on disk, what the model was saying when the run ended.
It costs nothing, it cannot embellish, and it is always available — even when the
provider is down and the model's summary never comes. RunResult carries both, side by
side, precisely so a reader can check the model's story against the harness's log.

What it looks like
------------------
    1. read_file(utils.py) -> ok
    2. edit_file(utils.py) -> ok
    3. run_bash(pytest -q) -> ERROR: 1 failed, 2 passed in 0.04s
    4. edit_file(utils.py) -> ERROR: old text not found in utils.py
    Last error: old text not found in utils.py. Closest match: ...
    Files written: utils.py
    Model's last words: The test still fails; let me look at the boundary again.
    5 model calls, 4 tool calls, 2 errors.

One line per tool call, in order, with the ONE argument that identifies the call (the
path, the command, the code...) so a reader can follow the story without the full
arguments; the outcome as a flat "ok" or the first line of the error; then the roll-up
lines that answer the questions a retry needs answered: what went wrong last, what
changed on disk, what the model was thinking. It is deliberately dry — a log, not an
explanation. The explanation is the model's job, and the digest is what keeps it honest.
"""

from __future__ import annotations

import json
import re

from cornac.core.messages import Message

# Which argument names a tool call, in order of preference. These are the builtin
# tools' primary arguments (read_file/write_file/edit_file/list_dir take `path`,
# run_bash `command`, run_python `code`, spawn_agent `task`, grep `pattern`,
# web_search `query`); a tool with none of them is shown by its first argument.
KEY_ARGS = ("path", "command", "code", "task", "pattern", "query")

# The tools whose successful calls change files. `Files written` is built from their
# `path` arguments, so a retry (or a human) knows what on disk is no longer pristine.
WRITING_TOOLS = ("write_file", "edit_file")

KEY_ARG_CHARS = 60      # a path or a command, not a whole file
ERROR_LINE_CHARS = 80   # the first line of an error, on the per-call line
LAST_ERROR_CHARS = 240  # the fuller excerpt on the "Last error:" line
LAST_WORDS_CHARS = 200  # the model's final text

# The loop appends its own notes to a tool result behind this marker on a fresh line
# (REPEAT_NOTE in core/agent.py). Only the loop can put the bare marker there — tool
# output has it rewritten to "[cornac?]" first — so cutting a result at this point
# gives back what the tool itself said, which is what an error excerpt should quote.
_HARNESS_NOTE = "\n[cornac] "

# How the built-in tools report a failure THEY handled. The registry sets is_error
# only when a tool raises (and the loop, for a veto or a denial); a tool that ran and
# found the request impossible answers in prose instead — "Error: `old` not found in
# utils.py" from edit_file, "Error: not a file: missing.py" from read_file — and
# run_bash/run_python prefix every result with the command's exit status. Both are
# failures to anyone reading the transcript, and the benchmark's own soft-error count
# (run_spike.py, n_soft_errors) already uses the "Error:" convention. A digest that
# read only is_error called five failed edits "ok" and listed the untouched file
# under "Files written" — the opposite of what happened, handed to the one reader
# the digest exists to keep honest.
_SOFT_ERROR_PREFIX = "Error:"
_EXIT_CODE = re.compile(r"\(exit code (-?\d+)\)")


def digest(messages: list[Message], since: int = 0) -> str:
    """The harness's own account of the run recorded in `messages[since:]`.

    `since` is the index of the run's user message: an Agent's message list spans
    every run() it has made, and the digest of one run must not recount another's.
    Tool results are matched to their calls per response (see _pair_calls), so a
    response that made several calls at once is listed one call per line, in the
    order the model made them. A call with no result at all — which the loop never
    produces, but a hand-built transcript might — is marked rather than skipped,
    since a missing result is itself something a reader should know about.

    A call is a failure when the loop flagged it (is_error: a raise, a veto, a
    denial) OR when the tool itself said so in its text — an "Error:" first line, or
    a non-zero "(exit code N)" from the shell tools. See _failed.

    Never raises on odd input (arguments that are not a dict, a result with no text):
    this runs at the moment a run has already gone wrong, and the account of a
    failure must not become a second failure.
    """
    run = messages[since:]

    model_calls = 0
    last_words: str | None = None
    for m in run:
        if m.role == "assistant":
            model_calls += 1
            if m.text and m.text.strip():
                last_words = m.text
    calls = _pair_calls(run)

    lines: list[str] = []
    errors: list[str] = []
    written: list[str] = []
    for i, (call, result) in enumerate(calls, 1):
        if result is None:
            outcome = "no result recorded"
        elif _failed(result):
            outcome = "ERROR: " + _failure_head(_tool_text(result), ERROR_LINE_CHARS)
            errors.append(_tool_text(result))
        else:
            outcome = "ok"
            args = call.arguments if isinstance(call.arguments, dict) else {}
            path = args.get("path")
            if call.name in WRITING_TOOLS and isinstance(path, str) and path:
                written.append(path)
        lines.append(f"{i}. {call.name}({_key_arg(call.arguments)}) -> {outcome}")

    if errors:
        lines.append("Last error: " + _excerpt(errors[-1], LAST_ERROR_CHARS))
    if written:
        # dict.fromkeys keeps first-seen order and drops duplicates: a file edited
        # three times is one file written.
        lines.append("Files written: " + ", ".join(dict.fromkeys(written)))
    if last_words is not None:
        lines.append("Model's last words: " + _squash(last_words, LAST_WORDS_CHARS))
    lines.append(f"{model_calls} model calls, {len(calls)} tool calls, {len(errors)} errors.")
    return "\n".join(lines)


def _pair_calls(run: list[Message]) -> list[tuple]:
    """(ToolCall, its tool Message or None) for every call in `run`, in the order made.

    The results that answer a response are the tool messages that immediately follow
    it — the loop appends them there, one per call, in the call's order — so pairing
    is done response by response. Within a response the ids are used when they line
    up one to one (a hand-built transcript may list results out of order); otherwise
    the results are taken by position.

    Why not one id -> result map over the whole run, which the first version used:
    ids are not unique across responses. Ollama does not send ids, so the provider
    synthesizes `call_{i}_{name}` with i the index WITHIN the response, and every
    turn's first write_file was `call_0_write_file`. Last-wins in a global map then
    gave every earlier call the last call's outcome: a run whose first edit worked
    and whose second failed showed two errors and no file written. The providers
    now number their synthesized ids across the whole conversation as well, but the
    digest must not depend on that — a transcript can come from anywhere.
    """
    pairs: list[tuple] = []
    i, n = 0, len(run)
    while i < n:
        m = run[i]
        i += 1
        if m.role != "assistant":
            continue
        block: list[Message] = []
        while i < n and run[i].role == "tool":
            block.append(run[i])
            i += 1
        ids = [call.id for call in m.tool_calls]
        by_id = {t.tool_call_id: t for t in block}
        if len(set(ids)) == len(ids) and len(by_id) == len(block) and all(c in by_id for c in ids):
            pairs.extend((call, by_id[call.id]) for call in m.tool_calls)
        else:
            pairs.extend(
                (call, block[k] if k < len(block) else None) for k, call in enumerate(m.tool_calls)
            )
    return pairs


def _failed(result: Message) -> bool:
    """True if this result reports a failure — flagged by the loop, or said by the tool."""
    if result.is_error:
        return True
    text = _tool_text(result).lstrip()
    if text.startswith(_SOFT_ERROR_PREFIX):
        return True
    exit_code = _EXIT_CODE.match(text)
    return exit_code is not None and exit_code.group(1) != "0"


def _failure_head(text: str, limit: int) -> str:
    """The first line of a failure, cut to `limit` — plus the next one after a bare exit code.

    "(exit code 127)" on its own says nothing about what went wrong; the shell's
    complaint is on the line after it, so the two are shown together.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    if len(lines) > 1 and _EXIT_CODE.fullmatch(lines[0]):
        return _cut(lines[0] + " " + lines[1], limit)
    return _cut(lines[0], limit)


def _tool_text(result: Message) -> str:
    """A tool result's own words: its text with any note the loop appended cut off."""
    return (result.text or "").split(_HARNESS_NOTE, 1)[0]


def _key_arg(arguments) -> str:
    """The one argument that identifies a call, flattened and cut to KEY_ARG_CHARS.

    The first of KEY_ARGS that is present wins, so `grep(pattern, path)` shows the
    path — the more specific of the two for a reader wondering where the model was
    looking — and `edit_file(path, old, new)` shows the path and not the edit. A
    tool with none of those shows its first argument; a call with no arguments at
    all shows nothing between the parentheses.
    """
    if not isinstance(arguments, dict):
        return _squash(str(arguments), KEY_ARG_CHARS)
    if not arguments:
        return ""
    for key in KEY_ARGS:
        if key in arguments:
            value = arguments[key]
            break
    else:
        value = next(iter(arguments.values()))
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return _squash(text, KEY_ARG_CHARS)


def _squash(text: str, limit: int) -> str:
    """Collapse all whitespace to single spaces and cut to `limit` characters.

    Code and command arguments span lines; the digest is one line per call.
    """
    return _cut(" ".join(text.split()), limit)


def _first_line(text: str, limit: int) -> str:
    """The first non-blank line of `text`, cut to `limit` characters."""
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return _cut(line, limit)


def _excerpt(text: str, limit: int) -> str:
    """A whitespace-squashed excerpt that keeps the head AND the tail of a long error.

    Cutting only the tail would lose the part that usually names the cause: pytest
    prints its FAILED lines last, and a traceback ends with the exception. The head
    stays too, because that is where a command's or tool's own first complaint is.
    """
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    head = limit // 2
    tail = limit - head - 5  # room for the " ... " between them
    return flat[:head] + " ... " + flat[-tail:]


def _cut(text: str, limit: int) -> str:
    """Cut `text` to at most `limit` characters, marking the cut with "..."."""
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)] + "..."
