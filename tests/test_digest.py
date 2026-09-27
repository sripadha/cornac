"""Tests for the digest (Week 4c): the harness's model-free account of an unfinished run.

The digest is built from a list of Messages and nothing else, so most tests hand it a
hand-built transcript and check the text line by line. One test at the end runs a
real Agent to max_steps and checks that RunResult.digest is that same account.
"""

import pytest

from cornac import Agent, Message, ToolRegistry
from cornac.core.digest import digest
from cornac.core.messages import ToolCall
from cornac.providers.base import Provider
from cornac.tools.base import tool


def _assistant(text="", *calls):
    return Message(role="assistant", text=text, tool_calls=list(calls))


def _call(call_id, name, **arguments):
    return ToolCall(call_id, name, arguments)


def _result(call_id, name, text, is_error=False):
    return Message(role="tool", text=text, tool_call_id=call_id, name=name, is_error=is_error)


def _user(text="go"):
    return Message.user(text)


# --- One line per tool call ----------------------------------------------------------------

def test_one_line_per_tool_call_in_order_then_the_counts():
    messages = [
        _user(),
        _assistant("", _call("1", "read_file", path="utils.py")),
        _result("1", "read_file", "def within(...): ..."),
        _assistant("", _call("2", "run_bash", command="pytest -q")),
        _result("2", "run_bash", "1 failed, 2 passed in 0.04s\nFAILED test_x", is_error=True),
    ]
    lines = digest(messages).splitlines()

    assert lines[0] == "1. read_file(utils.py) -> ok"
    assert lines[1] == "2. run_bash(pytest -q) -> ERROR: 1 failed, 2 passed in 0.04s"
    assert lines[2].startswith("Last error: ")
    assert lines[-1] == "2 model calls, 2 tool calls, 1 errors."


def test_several_calls_in_one_response_are_listed_in_order_by_their_ids():
    messages = [
        _user(),
        _assistant("", _call("a", "read_file", path="a.py"), _call("b", "read_file", path="b.py")),
        _result("b", "read_file", "B"),       # results may arrive in any order
        _result("a", "read_file", "A"),
    ]
    lines = digest(messages).splitlines()
    assert lines[:2] == ["1. read_file(a.py) -> ok", "2. read_file(b.py) -> ok"]
    assert lines[-1] == "1 model calls, 2 tool calls, 0 errors."


def test_a_call_with_no_result_is_marked_not_skipped():
    messages = [_user(), _assistant("", _call("1", "mark", label="x"))]
    lines = digest(messages).splitlines()
    assert lines[0] == "1. mark(x) -> no result recorded"


# --- The key argument --------------------------------------------------------------------------

@pytest.mark.parametrize("arguments, shown", [
    ({"path": "a.py", "content": "print(1)"}, "a.py"),          # path over content
    ({"pattern": "def ", "path": "src"}, "src"),                # path over pattern
    ({"command": "pytest -q"}, "pytest -q"),
    ({"code": "print(1)"}, "print(1)"),
    ({"task": "count the files"}, "count the files"),
    ({"pattern": "TODO"}, "TODO"),
    ({"query": "python dataclass"}, "python dataclass"),
    ({"label": "x", "other": "y"}, "x"),                        # none present: the first
    ({}, ""),                                                   # no arguments at all
    ({"max_results": 5}, "5"),                                  # a non-string is shown too
])
def test_key_arg_is_the_first_of_path_command_code_task_pattern_query_else_the_first(arguments, shown):
    messages = [_user(), _assistant("", ToolCall("1", "t", arguments)), _result("1", "t", "ok")]
    assert digest(messages).splitlines()[0] == f"1. t({shown}) -> ok"


def test_key_arg_is_flattened_and_cut_to_sixty_chars():
    code = "import os\n\nfor f in os.listdir('.'):\n    print(f)\n" + "x = 1\n" * 30
    messages = [_user(), _assistant("", _call("1", "run_python", code=code)), _result("1", "run_python", "")]
    line = digest(messages).splitlines()[0]

    inside = line[len("1. run_python("):line.index(") -> ok")]
    assert "\n" not in inside
    assert inside.startswith("import os for f in os.listdir('.'): print(f)")
    assert inside.endswith("...")
    assert len(inside) == 60


def test_key_arg_survives_arguments_that_are_not_a_dict():
    messages = [_user(), _assistant("", ToolCall("1", "t", "not a dict")), _result("1", "t", "ok")]
    assert digest(messages).splitlines()[0] == "1. t(not a dict) -> ok"


# --- Errors -----------------------------------------------------------------------------------------

def test_error_line_shows_the_first_non_blank_line_cut_to_eighty_chars():
    long_first = "E" * 100
    messages = [
        _user(),
        _assistant("", _call("1", "run_bash", command="x")),
        _result("1", "run_bash", "\n\n" + long_first + "\nsecond line", is_error=True),
    ]
    line = digest(messages).splitlines()[0]
    shown = line[len("1. run_bash(x) -> ERROR: "):]
    assert shown == "E" * 77 + "..."
    assert len(shown) == 80


def test_last_error_is_the_most_recent_one_with_head_and_tail():
    early = "first failure"
    late = "HEAD " + "m" * 400 + " TAIL: AssertionError"
    messages = [
        _user(),
        _assistant("", _call("1", "run_bash", command="a")),
        _result("1", "run_bash", early, is_error=True),
        _assistant("", _call("2", "run_bash", command="b")),
        _result("2", "run_bash", late, is_error=True),
    ]
    last = next(ln for ln in digest(messages).splitlines() if ln.startswith("Last error: "))
    assert "first failure" not in last
    assert last.startswith("Last error: HEAD mmm")
    assert last.endswith("TAIL: AssertionError")           # the end is where the cause is
    assert " ... " in last
    assert len(last) <= len("Last error: ") + 240


def test_no_errors_means_no_last_error_line():
    messages = [_user(), _assistant("", _call("1", "mark", label="x")), _result("1", "mark", "ok")]
    assert "Last error" not in digest(messages)
    assert digest(messages).splitlines()[-1] == "1 model calls, 1 tool calls, 0 errors."


def test_the_loops_own_repeat_note_is_cut_off_an_error_excerpt():
    # The loop appends "[cornac] This exact call was already made..." to a repeated
    # result. That is the harness talking, not the tool; the digest quotes the tool.
    text = "old text not found in utils.py\n[cornac] This exact call was already made 2 times in this run."
    messages = [_user(), _assistant("", _call("1", "edit_file", path="utils.py")),
                _result("1", "edit_file", text, is_error=True)]
    out = digest(messages)
    assert "Last error: old text not found in utils.py" in out
    assert "already made" not in out


# --- Files written -------------------------------------------------------------------------------

def test_files_written_lists_ok_write_and_edit_paths_once_and_skips_failed_ones():
    messages = [
        _user(),
        _assistant("", _call("1", "write_file", path="a.py", content="x")),
        _result("1", "write_file", "wrote a.py"),
        _assistant("", _call("2", "edit_file", path="b.py", old="1", new="2")),
        _result("2", "edit_file", "edited b.py"),
        _assistant("", _call("3", "edit_file", path="a.py", old="1", new="2")),
        _result("3", "edit_file", "edited a.py"),                        # a.py again: once
        _assistant("", _call("4", "edit_file", path="c.py", old="1", new="2")),
        _result("4", "edit_file", "old text not found", is_error=True),  # failed: not written
        _assistant("", _call("5", "read_file", path="d.py")),
        _result("5", "read_file", "..."),                                # reading is not writing
    ]
    lines = digest(messages).splitlines()
    assert "Files written: a.py, b.py" in lines
    assert lines.index("Files written: a.py, b.py") > lines.index("Last error: old text not found")


def test_no_writes_means_no_files_written_line():
    messages = [_user(), _assistant("", _call("1", "read_file", path="a.py")), _result("1", "read_file", "x")]
    assert "Files written" not in digest(messages)


# --- The model's last words ----------------------------------------------------------------------

def test_models_last_words_are_the_last_non_blank_assistant_text_cut_to_two_hundred():
    long = "The boundary check is wrong. " * 20   # well over 200 chars
    messages = [
        _user(),
        _assistant("I will look at the file.", _call("1", "read_file", path="a.py")),
        _result("1", "read_file", "..."),
        _assistant(long, _call("2", "edit_file", path="a.py", old="<", new="<=")),
        _result("2", "edit_file", "edited"),
        _assistant("", _call("3", "run_bash", command="pytest")),      # blank: not last words
        _result("3", "run_bash", "ok"),
    ]
    lines = digest(messages).splitlines()
    words = next(ln for ln in lines if ln.startswith("Model's last words: "))
    shown = words[len("Model's last words: "):]
    assert shown.startswith("The boundary check is wrong.")
    assert shown.endswith("...")
    assert len(shown) == 200
    assert "I will look" not in words
    assert lines.index(words) < len(lines) - 1        # the counts line is still last


def test_no_assistant_text_means_no_last_words_line():
    messages = [_user(), _assistant("", _call("1", "mark", label="x")), _result("1", "mark", "ok")]
    assert "Model's last words" not in digest(messages)


# --- since, and the empty case ------------------------------------------------------------------

def test_since_limits_the_digest_to_this_run():
    earlier_run = [
        _user("first"),
        _assistant("", _call("old", "read_file", path="old.py")),
        _result("old", "read_file", "..."),
        _assistant("first answer"),
    ]
    this_run = [
        _user("second"),
        _assistant("", _call("new", "run_bash", command="pytest")),
        _result("new", "run_bash", "boom", is_error=True),
    ]
    out = digest(earlier_run + this_run, since=len(earlier_run))
    assert "old.py" not in out and "first answer" not in out
    assert out.splitlines()[0] == "1. run_bash(pytest) -> ERROR: boom"
    assert out.splitlines()[-1] == "1 model calls, 1 tool calls, 1 errors."


def test_an_empty_run_is_just_the_counts_line():
    assert digest([]) == "0 model calls, 0 tool calls, 0 errors."
    assert digest([_user()]) == "0 model calls, 0 tool calls, 0 errors."


def test_a_run_with_only_text_has_last_words_and_counts():
    out = digest([_user(), _assistant("Let me think.")])
    assert out == "Model's last words: Let me think.\n1 model calls, 0 tool calls, 0 errors."


# --- From a real run -----------------------------------------------------------------------------

ran: list[str] = []


@tool()
def mark(label: str = "x") -> str:
    """Record that this tool ran."""
    ran.append(label)
    return f"marked {label}"


@tool()
def fail(why: str = "") -> str:
    """A tool that always raises, so the registry turns it into an error result."""
    raise RuntimeError(why or "it broke")


class ScriptedProvider(Provider):
    def __init__(self, script):
        self.script = list(script)

    def complete(self, messages, tools):
        return self.script.pop(0)


def test_run_result_digest_is_the_digest_of_this_run():
    ran.clear()
    script = [
        Message(role="assistant", text="Marking.", tool_calls=[ToolCall("c1", "mark", {"label": "a"})]),
        Message(role="assistant", tool_calls=[ToolCall("c2", "fail", {"why": "disk full"})]),
        Message(role="assistant", text="Still going.", tool_calls=[ToolCall("c3", "mark", {"label": "b"})]),
    ]
    agent = Agent(provider=ScriptedProvider(script), registry=ToolRegistry([mark, fail]),
                  max_steps=3, wrap_up=False)
    result = agent.run("go")

    assert result.stop_reason == "max_steps"
    assert result.digest.splitlines() == [
        "1. mark(a) -> ok",
        "2. fail(disk full) -> ERROR: RuntimeError: disk full",
        "3. mark(b) -> ok",
        "Last error: RuntimeError: disk full",
        "Model's last words: Still going.",
        "3 model calls, 3 tool calls, 1 errors.",
    ]
    # A second run on the same agent digests only itself.
    agent.provider.script = [Message(role="assistant", tool_calls=[ToolCall("c4", "mark", {"label": "c"})])]
    agent.max_steps = 1
    second = agent.run("again")
    assert second.digest.splitlines() == ["1. mark(c) -> ok", "1 model calls, 1 tool calls, 0 errors."]


# --- Pairing results to calls ---------------------------------------------------------------------
#
# Ollama does not send tool-call ids, so the provider makes them up. Until Week 4c's
# fix it restarted at 0 on every reply, so every turn's first write_file was
# `call_0_write_file`; a digest that paired by id over the whole run gave each earlier
# call the LAST call's outcome. Results are now paired per response — the tool
# messages right after an assistant turn answer that turn — and ids only settle the
# order within one response.

def test_results_are_paired_per_response_so_a_reused_id_cannot_cross_turns():
    same_id = "call_0_write_file"     # what two consecutive Ollama turns used to get
    messages = [
        _user(),
        _assistant("", _call(same_id, "write_file", path="a.py", content="x")),
        _result(same_id, "write_file", "wrote a.py"),
        _assistant("", _call(same_id, "write_file", path="b.py", content="y")),
        _result(same_id, "write_file", "Error: SyntaxError in b.py", is_error=True),
    ]
    lines = digest(messages).splitlines()

    assert lines[0] == "1. write_file(a.py) -> ok"
    assert lines[1] == "2. write_file(b.py) -> ERROR: Error: SyntaxError in b.py"
    assert "Files written: a.py" in lines                    # the good write is not lost
    assert lines[-1] == "2 model calls, 2 tool calls, 1 errors."


def test_within_one_response_results_out_of_order_are_still_matched_by_id():
    messages = [
        _user(),
        _assistant("", _call("a", "read_file", path="a.py"), _call("b", "run_bash", command="x")),
        _result("b", "run_bash", "Error: boom"),
        _result("a", "read_file", "A"),
    ]
    lines = digest(messages).splitlines()
    assert lines[:2] == ["1. read_file(a.py) -> ok", "2. run_bash(x) -> ERROR: Error: boom"]


def test_when_ids_do_not_line_up_results_are_taken_by_position():
    # A hand-built transcript with ids that match nothing: position is the best guess,
    # and a call beyond the results it got is marked, not paired with another turn's.
    messages = [
        _user(),
        _assistant("", _call("x", "mark", label="1"), _call("y", "mark", label="2")),
        _result("?", "mark", "first"),
        _assistant("", _call("z", "mark", label="3")),
        _result("??", "mark", "Error: third", is_error=False),
    ]
    lines = digest(messages).splitlines()
    assert lines[:3] == [
        "1. mark(1) -> ok",
        "2. mark(2) -> no result recorded",
        "3. mark(3) -> ERROR: Error: third",
    ]


# --- Soft failures: what the tool said, not only what the loop flagged ------------------------------
#
# The registry sets is_error only when a tool raises; every built-in reports the
# failures it handles as text — "Error: ..." from the file tools, "(exit code N)" from
# the shell tools. The digest must read those too, or five failed edits read "ok" and
# the untouched file is listed as written.

def test_error_prefixed_results_and_non_zero_exit_codes_are_failures():
    messages = [
        _user(),
        _assistant("", _call("1", "edit_file", path="utils.py", old="a", new="b")),
        _result("1", "edit_file", "Error: `old` not found in utils.py. Closest match: ..."),
        _assistant("", _call("2", "run_bash", command="pytest -q")),
        _result("2", "run_bash", "(exit code 1)\n============ 1 failed, 2 passed in 0.04s"),
        _assistant("", _call("3", "read_file", path="missing.py")),
        _result("3", "read_file", "Error: not a file: missing.py"),
        _assistant("", _call("4", "run_bash", command="ls")),
        _result("4", "run_bash", "(exit code 0)\na.py"),                 # zero: fine
        _assistant("", _call("5", "read_file", path="notes.md")),
        _result("5", "read_file", "Errors are documented in section 3."),   # prose, not a failure
    ]
    lines = digest(messages).splitlines()

    assert lines[0] == "1. edit_file(utils.py) -> ERROR: Error: `old` not found in utils.py. Closest match: ..."
    assert lines[1] == "2. run_bash(pytest -q) -> ERROR: (exit code 1) ============ 1 failed, 2 passed in 0.04s"
    assert lines[2] == "3. read_file(missing.py) -> ERROR: Error: not a file: missing.py"
    assert lines[3] == "4. run_bash(ls) -> ok"
    assert lines[4] == "5. read_file(notes.md) -> ok"
    assert "Last error: Error: not a file: missing.py" in lines
    assert not any(ln.startswith("Files written") for ln in lines)       # the failed edit wrote nothing
    assert lines[-1] == "5 model calls, 5 tool calls, 3 errors."


def test_a_bare_exit_code_line_is_joined_with_the_shells_complaint():
    messages = [
        _user(),
        _assistant("", _call("1", "run_bash", command="definitely_not_a_command")),
        _result("1", "run_bash", "(exit code 127)\n/bin/sh: 1: definitely_not_a_command: not found"),
    ]
    line = digest(messages).splitlines()[0]
    assert line == (
        "1. run_bash(definitely_not_a_command) -> ERROR: (exit code 127) "
        "/bin/sh: 1: definitely_not_a_command: not found"
    )


def test_the_real_builtins_soft_failures_reach_the_digest_through_an_agent(tmp_path):
    # The finding's reproduction: the real edit_file, run_bash and read_file on a
    # temp workspace, each failing the way it really does (is_error stays False),
    # and the RunResult's digest must say so.
    from cornac.tools.builtin import default_tools
    from cornac.tools.workspace import Workspace

    (tmp_path / "utils.py").write_text("def f():\n    return 1\n")
    before = (tmp_path / "utils.py").read_text()
    script = [
        Message(role="assistant", tool_calls=[ToolCall("c1", "edit_file", {
            "path": "utils.py", "old": "THIS TEXT IS NOT THERE", "new": "x"})]),
        Message(role="assistant", tool_calls=[ToolCall("c2", "run_bash", {
            "command": "definitely_not_a_command_xyz"})]),
        Message(role="assistant", tool_calls=[ToolCall("c3", "read_file", {"path": "missing.py"})]),
    ]
    agent = Agent(provider=ScriptedProvider(script), registry=ToolRegistry(default_tools(Workspace(tmp_path))),
                  max_steps=3, wrap_up=False)

    result = agent.run("go")

    assert result.stop_reason == "max_steps"
    assert all(not m.is_error for m in agent.messages if m.role == "tool")   # the premise: soft failures
    lines = result.digest.splitlines()
    assert lines[0].startswith("1. edit_file(utils.py) -> ERROR: Error: `old` not found in utils.py")
    assert lines[1].startswith("2. run_bash(definitely_not_a_command_xyz) -> ERROR: (exit code 127)")
    assert lines[2] == "3. read_file(missing.py) -> ERROR: Error: not a file: missing.py"
    assert any(ln.startswith("Last error: Error: not a file: missing.py") for ln in lines)
    assert not any(ln.startswith("Files written") for ln in lines)
    assert lines[-1] == "3 model calls, 3 tool calls, 3 errors."
    assert (tmp_path / "utils.py").read_text() == before                    # nothing was written
