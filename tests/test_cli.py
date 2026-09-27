"""Tests for the command line (Week 4c): cornac "task" / python -m cornac.

The CLI is glue, and glue is tested by checking that each piece was wired to the
right neighbour: the flags reach the Namespace, the Namespace reaches the Agent (the
right policy, approver, prompt, tool set), the RunResult reaches the screen in the
agreed shape, and the outcome reaches the shell as the agreed exit code. A scripted
provider plays the model, so every run here is offline and instant; the tools, the
fence and the permission gate are the real ones.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

import cornac.cli as cli
from cornac.cli import CLI_POLICY, UsageError, build_agent, main, make_provider, parse_args, print_result
from cornac.core.messages import Message, ToolCall, Usage
from cornac.core.result import RunResult
from cornac.permissions.policy import Decision
from cornac.permissions.prompt import cli_ask
from cornac.prompts import DEFAULT_PERSONA
from cornac.providers.base import Provider
from cornac.providers.ollama import DEFAULT_MODEL, OllamaProvider
from cornac.tools.workspace import WorkspaceError

REPO_ROOT = Path(__file__).resolve().parent.parent

# Every built-in the CLI hands the model, by name — pinned so a tool dropped from
# default_tools() (or a renamed one) shows up here and not in a confused user's run.
DEFAULT_TOOL_NAMES = {
    "read_file", "write_file", "edit_file", "list_dir", "grep", "run_bash", "run_python",
    "web_search", "web_fetch", "get_current_time", "spawn_agent",
}


class ScriptedProvider(Provider):
    """Plays the model: one pre-baked reply per call, recorded for the test to inspect.

    Once the script runs out it answers with plain text. That matters for a run that
    hits max_steps: the Week 4c-A wrap-up makes one more call (tools switched off)
    for the model's summary, and this stand-in must answer it rather than raise.
    """

    def __init__(self, *script: Message):
        self._script = list(script)
        self.calls: list[tuple[list[Message], list[dict]]] = []

    def complete(self, messages, tools):
        self.calls.append((list(messages), list(tools)))
        if self._script:
            return self._script.pop(0)
        return Message(role="assistant", text="Wrap-up: I only listed the directory.",
                       usage=Usage(5, 5))


def _tool(name: str, **arguments) -> Message:
    return Message(role="assistant", tool_calls=[ToolCall("c1", name, arguments)], usage=Usage(10, 2))


def _final(text: str = "All done.") -> Message:
    return Message(role="assistant", text=text, usage=Usage(10, 3))


def _use(monkeypatch, provider) -> None:
    """Make the CLI build its agent around `provider` instead of a real backend."""
    monkeypatch.setattr(cli, "make_provider", lambda args: provider)


# --- parse_args -------------------------------------------------------------------------

def test_defaults():
    args = parse_args(["fix the bug"])
    assert args.task == "fix the bug"
    assert args.provider == "ollama" and args.model is None and args.base_url is None
    assert args.workspace == "." and args.policy is None
    assert args.max_steps == 20 and args.transcript is None
    assert args.no_instructions is False and args.quiet is False


def test_every_flag_reaches_the_namespace():
    args = parse_args([
        "t", "--provider", "anthropic", "--model", "m", "--base-url", "http://h:1",
        "--workspace", "/w", "--policy", "p.yaml", "--max-steps", "3",
        "--transcript", "t.jsonl", "--no-instructions", "--quiet",
    ])
    assert (args.provider, args.model, args.base_url) == ("anthropic", "m", "http://h:1")
    assert (args.workspace, args.policy, args.max_steps) == ("/w", "p.yaml", 3)
    assert args.transcript == "t.jsonl" and args.no_instructions and args.quiet


def test_task_file_is_read_into_task(tmp_path):
    f = tmp_path / "task.md"
    f.write_text("# Task\n\nRename foo to bar.\n")
    assert parse_args(["--task-file", str(f)]).task == "# Task\n\nRename foo to bar.\n"


@pytest.mark.parametrize("argv, fragment", [
    ([], "no task"),
    (["   "], "no task"),
    (["--task-file", "/nonexistent/task.md"], "cannot read task file"),
    (["--provider", "gpt", "t"], "invalid choice"),
    (["t", "--max-steps", "0"], "at least 1"),
])
def test_usage_errors_raise_instead_of_exiting(argv, fragment):
    # argparse would sys.exit(2), and 2 means "did not finish" here; see UsageError.
    with pytest.raises(UsageError, match=fragment):
        parse_args(argv)


def test_task_and_task_file_together_is_a_usage_error(tmp_path):
    f = tmp_path / "t.md"
    f.write_text("x")
    with pytest.raises(UsageError, match="not both"):
        parse_args(["also this", "--task-file", str(f)])


# --- make_provider ----------------------------------------------------------------------

def test_ollama_is_the_default_provider():
    provider = make_provider(parse_args(["t"]))
    assert isinstance(provider, OllamaProvider)
    assert provider.model == DEFAULT_MODEL


def test_ollama_takes_model_and_host():
    provider = make_provider(parse_args(["t", "--model", "qwen3:8b", "--base-url", "http://h:1"]))
    assert provider.model == "qwen3:8b" and provider.host == "http://h:1"


def _fake_module(monkeypatch, module: str, cls_name: str, recorded: dict):
    """Stand a recording class in for a provider module (no SDK, no network)."""
    class Fake:
        def __init__(self, **kwargs):
            recorded.update(kwargs)
    mod = types.ModuleType(module)
    setattr(mod, cls_name, Fake)
    monkeypatch.setitem(sys.modules, module, mod)


def test_anthropic_gets_only_the_model_flag(monkeypatch):
    recorded: dict = {}
    _fake_module(monkeypatch, "cornac.providers.anthropic", "AnthropicProvider", recorded)
    make_provider(parse_args(["t", "--provider", "anthropic"]))
    assert recorded == {}  # the provider's own default model
    make_provider(parse_args(["t", "--provider", "anthropic", "--model", "claude-x"]))
    assert recorded == {"model": "claude-x"}


def test_openai_gets_model_and_base_url(monkeypatch):
    recorded: dict = {}
    _fake_module(monkeypatch, "cornac.providers.openai_compat", "OpenAICompatibleProvider", recorded)
    make_provider(parse_args(["t", "--provider", "openai"]))
    assert recorded == {"model": ""}
    make_provider(parse_args(["t", "--provider", "openai", "--model", "m", "--base-url", "http://v:8000/v1"]))
    assert recorded == {"model": "m", "base_url": "http://v:8000/v1"}


# --- build_agent ------------------------------------------------------------------------

def test_build_agent_wires_the_pieces(tmp_path):
    provider = ScriptedProvider()
    agent = build_agent(parse_args(["t", "--workspace", str(tmp_path), "--max-steps", "7"]), provider)
    assert agent.provider is provider
    assert agent.max_steps == 7
    assert agent.approver is cli_ask
    assert set(agent.registry.names()) == DEFAULT_TOOL_NAMES
    assert agent.system_prompt.startswith(DEFAULT_PERSONA)
    assert str(tmp_path.resolve()) in agent.system_prompt


def test_build_agent_falls_back_to_the_factory(monkeypatch, tmp_path):
    provider = ScriptedProvider()
    _use(monkeypatch, provider)
    assert build_agent(parse_args(["t", "--workspace", str(tmp_path)])).provider is provider


def test_default_policy_allows_read_only_tools_and_asks_about_the_rest(tmp_path):
    agent = build_agent(parse_args(["t", "--workspace", str(tmp_path)]), ScriptedProvider())
    for name in ("read_file", "list_dir", "grep", "get_current_time"):
        assert agent.policy.check(ToolCall("1", name, {})) == Decision.ALLOW, name
    for name in ("write_file", "edit_file", "run_bash", "run_python", "web_search",
                 "web_fetch", "spawn_agent", "something_new"):
        assert agent.policy.check(ToolCall("1", name, {})) == Decision.ASK, name
    assert set(CLI_POLICY["tools"]) == {"read_file", "list_dir", "grep", "get_current_time", "run_bash"}
    # run_bash is listed for its deny list only: it still asks by default, and the
    # ordinary command asks too.
    assert agent.policy.check(ToolCall("1", "run_bash", {"command": "pytest -q"})) == Decision.ASK


def test_the_default_policy_carries_the_librarys_run_bash_deny_list():
    from cornac.permissions.policy import DEFAULT_RULES

    assert CLI_POLICY["tools"]["run_bash"] == DEFAULT_RULES["tools"]["run_bash"]
    assert CLI_POLICY["tools"]["run_bash"] is not DEFAULT_RULES["tools"]["run_bash"]   # a copy
    policy = build_agent(parse_args(["t"]), ScriptedProvider()).policy
    for command in ("sudo rm -rf ~", "rm -rf /", "mkfs.ext4 /dev/sda", "dd if=/dev/zero of=/dev/sda"):
        assert policy.check(ToolCall("1", "run_bash", {"command": command})) == Decision.DENY, command


def test_an_always_on_run_bash_keeps_the_deny_list_in_force():
    # The finding: under the old CLI_POLICY, one "always" on `pytest -q` turned
    # `sudo rm -rf ~` into an auto-allowed call for the rest of the session — while
    # cli_ask told the user "its deny patterns still apply". Now that is true.
    from cornac.permissions.policy import Policy

    policy = Policy(CLI_POLICY)
    policy.remember("run_bash", Decision.ALLOW)

    assert policy.check(ToolCall("1", "run_bash", {"command": "pytest -q"})) == Decision.ALLOW
    assert policy.check(ToolCall("2", "run_bash", {"command": "sudo rm -rf ~"})) == Decision.DENY


def test_always_answered_at_the_prompt_does_not_let_a_denied_command_run(monkeypatch, tmp_path, capsys):
    # End to end through main(): the first run_bash prompts, the user answers "a",
    # and the second run_bash — a denied pattern — is refused without a prompt and
    # without running. The model sees a permission error, not the command's output.
    monkeypatch.setattr("builtins.input", lambda prompt="": "a")
    marker = tmp_path / "ran-the-denied-command"
    provider = ScriptedProvider(
        _tool("run_bash", command="echo hello"),
        _tool("run_bash", command=f"sudo touch {marker}"),
        _final("ok"),
    )
    _use(monkeypatch, provider)

    assert main(["t", "--workspace", str(tmp_path)]) == 0

    out = capsys.readouterr().out
    assert out.count("cornac wants to run a tool") == 1            # asked once, then remembered
    assert "its deny patterns still apply" in out                   # ...and that promise holds:
    first, second = provider.calls[1][0][-1], provider.calls[2][0][-1]
    assert first.role == "tool" and not first.is_error and "hello" in first.text
    assert second.role == "tool" and second.is_error and "Permission denied" in second.text
    assert not marker.exists()


def test_a_non_utf8_task_file_is_a_usage_error_not_a_traceback(tmp_path, capsys):
    bad = tmp_path / "task.md"
    bad.write_bytes(b"\xff\xfe not utf-8")

    with pytest.raises(UsageError, match="cannot read task file"):
        parse_args(["--task-file", str(bad)])

    assert main(["--task-file", str(bad), "--workspace", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "cannot read task file" in captured.err
    assert "Traceback" not in captured.err


def test_policy_file_replaces_the_default(tmp_path):
    (tmp_path / "p.yaml").write_text("default: deny\ntools:\n  read_file: allow\n")
    agent = build_agent(
        parse_args(["t", "--workspace", str(tmp_path), "--policy", str(tmp_path / "p.yaml")]),
        ScriptedProvider(),
    )
    assert agent.policy.check(ToolCall("1", "read_file", {})) == Decision.ALLOW
    assert agent.policy.check(ToolCall("1", "write_file", {})) == Decision.DENY


def test_agents_md_is_in_the_prompt_unless_disabled(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Use tabs.")
    with_file = build_agent(parse_args(["t", "--workspace", str(tmp_path)]), ScriptedProvider())
    assert "Use tabs." in with_file.system_prompt
    without = build_agent(
        parse_args(["t", "--workspace", str(tmp_path), "--no-instructions"]), ScriptedProvider()
    )
    assert "Use tabs." not in without.system_prompt


def test_workspace_must_be_a_directory(tmp_path):
    with pytest.raises(WorkspaceError):
        build_agent(parse_args(["t", "--workspace", str(tmp_path / "nope")]), ScriptedProvider())


# --- print_result -----------------------------------------------------------------------

def _result(**over) -> RunResult:
    base = dict(text="42", stop_reason="done", steps=3, duration=4.24, usage=Usage(1000, 234))
    base.update(over)
    return RunResult(**base)


def _printed(result, quiet=False) -> str:
    out = io.StringIO()
    print_result(result, quiet=quiet, out=out)
    return out.getvalue()


def test_done_prints_the_text_then_one_footer_line():
    assert _printed(_result()) == "42\nstop=done steps=3 tokens=1234 time=4.2s\n"


def test_quiet_drops_the_footer():
    assert _printed(_result(), quiet=True) == "42\n"


def test_unfinished_prints_header_digest_and_labelled_summary():
    out = _printed(_result(text=None, stop_reason="max_steps", steps=10,
                           digest="1. read_file(a.py) -> ok", summary="I read a.py."))
    lines = out.splitlines()
    assert lines[0] == "Did not finish (max_steps)."          # always the first line
    assert "1. read_file(a.py) -> ok" in out
    assert "I read a.py." in out
    assert "not an answer" in out                              # the summary is labelled
    assert out.index("Did not finish") < out.index("1. read_file") < out.index("I read a.py.")
    assert lines[-1].startswith("stop=max_steps steps=10 ")


def test_unfinished_without_digest_or_summary_is_header_and_footer_only():
    out = _printed(_result(text=None, stop_reason="stuck", steps=4))
    assert out.splitlines() == ["Did not finish (stuck).", "stop=stuck steps=4 tokens=1234 time=4.2s"]


# --- main ---------------------------------------------------------------------------------

def test_done_run_returns_zero_and_prints_answer_and_footer(monkeypatch, tmp_path, capsys):
    (tmp_path / "a.txt").write_text("a")
    provider = ScriptedProvider(_tool("list_dir", path="."), _final("There is one file: a.txt."))
    _use(monkeypatch, provider)

    rc = main(["list the files", "--workspace", str(tmp_path)])

    out = capsys.readouterr().out
    assert rc == 0
    assert "There is one file: a.txt." in out
    assert "stop=done steps=2 tokens=25 time=" in out
    # What the model actually saw: the assembled prompt, the task, the full tool set.
    first_messages, first_tools = provider.calls[0]
    assert first_messages[0].role == "system" and first_messages[0].text.startswith(DEFAULT_PERSONA)
    assert first_messages[1].role == "user" and first_messages[1].text == "list the files"
    assert {t["name"] for t in first_tools} == DEFAULT_TOOL_NAMES
    # The tool ran for real, inside the fence: its result went back to the model.
    assert "a.txt" in provider.calls[1][0][-1].text


def test_quiet_run_prints_only_the_answer(monkeypatch, tmp_path, capsys):
    _use(monkeypatch, ScriptedProvider(_final("Just this.")))
    assert main(["t", "--workspace", str(tmp_path), "--quiet"]) == 0
    assert capsys.readouterr().out == "Just this.\n"


def test_task_file_reaches_the_model(monkeypatch, tmp_path, capsys):
    (tmp_path / "task.md").write_text("Do the thing described here.\n")
    provider = ScriptedProvider(_final())
    _use(monkeypatch, provider)
    assert main(["--task-file", str(tmp_path / "task.md"), "--workspace", str(tmp_path)]) == 0
    assert provider.calls[0][0][1].text == "Do the thing described here.\n"


def test_unfinished_run_returns_two_and_says_so(monkeypatch, tmp_path, capsys):
    # One step allowed, and the model spends it on a tool call: max_steps, no answer.
    _use(monkeypatch, ScriptedProvider(_tool("list_dir", path=".")))
    rc = main(["t", "--workspace", str(tmp_path), "--max-steps", "1"])
    out = capsys.readouterr().out
    assert rc == 2
    assert out.splitlines()[0] == "Did not finish (max_steps)."
    assert "stop=max_steps steps=1 " in out
    # The unfinished run carries both layers: the harness digest first, then the
    # model's own account under its "not an answer" label, never bare.
    assert out.index("What happened (harness digest):") < out.index("1. list_dir(.)")
    assert out.index("1. list_dir(.)") < out.index("not an answer")
    assert out.index("not an answer") < out.index("Wrap-up: I only listed the directory.")


def test_a_write_is_asked_about_and_no_refuses_it(monkeypatch, tmp_path, capsys):
    # The default policy says ASK for write_file, and cli_ask is the approver: the
    # human's "n" becomes a permission error for the model, and no file appears.
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    provider = ScriptedProvider(_tool("write_file", path="new.txt", content="hi"), _final("ok"))
    _use(monkeypatch, provider)
    assert main(["t", "--workspace", str(tmp_path)]) == 0
    assert not (tmp_path / "new.txt").exists()
    tool_msg = provider.calls[1][0][-1]
    assert tool_msg.role == "tool" and tool_msg.is_error and "Permission denied" in tool_msg.text
    assert "cornac wants to run a tool" in capsys.readouterr().out


def test_a_write_approved_with_yes_runs(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    _use(monkeypatch, ScriptedProvider(_tool("write_file", path="new.txt", content="hi"), _final("ok")))
    assert main(["t", "--workspace", str(tmp_path)]) == 0
    assert (tmp_path / "new.txt").read_text() == "hi"


def test_usage_error_returns_one_with_a_message_on_stderr(capsys):
    assert main([]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "no task" in captured.err and "usage: cornac" in captured.err


def test_provider_construction_error_returns_one(monkeypatch, tmp_path, capsys):
    def boom(args):
        raise RuntimeError("Ollama daemon is not running")
    monkeypatch.setattr(cli, "make_provider", boom)
    assert main(["t", "--workspace", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "cornac: RuntimeError: Ollama daemon is not running" in captured.err


def test_error_during_the_run_returns_one(monkeypatch, tmp_path, capsys):
    class Failing(Provider):
        def complete(self, messages, tools):
            raise ConnectionError("connection refused")
    _use(monkeypatch, Failing())
    assert main(["t", "--workspace", str(tmp_path)]) == 1
    assert "ConnectionError: connection refused" in capsys.readouterr().err


def test_bad_workspace_returns_one(monkeypatch, tmp_path, capsys):
    _use(monkeypatch, ScriptedProvider(_final()))
    assert main(["t", "--workspace", str(tmp_path / "missing")]) == 1
    assert "WorkspaceError" in capsys.readouterr().err


def test_ctrl_c_returns_130(monkeypatch, tmp_path, capsys):
    class Interrupted(Provider):
        def complete(self, messages, tools):
            raise KeyboardInterrupt
    _use(monkeypatch, Interrupted())
    assert main(["t", "--workspace", str(tmp_path)]) == 130
    assert "interrupted" in capsys.readouterr().err


def test_transcript_flag_writes_jsonl(monkeypatch, tmp_path):
    pytest.importorskip("cornac.transcript")
    (tmp_path / "a.txt").write_text("a")
    _use(monkeypatch, ScriptedProvider(_tool("list_dir", path="."), _final()))
    path = tmp_path / "run.jsonl"
    assert main(["t", "--workspace", str(tmp_path), "--transcript", str(path), "--quiet"]) == 0
    lines = path.read_text().splitlines()
    records = [json.loads(line) for line in lines]          # every line is one JSON object
    assert records and all("event" in r for r in records)
    assert records[0]["event"] == "run_config"               # attach() writes it first
    assert any(r["event"] == "post_tool_use" for r in records)


# --- the two entry points -------------------------------------------------------------------

def _run_module(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "cornac", *argv], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=60)


def test_python_m_cornac_help_exits_zero():
    proc = _run_module("--help")
    assert proc.returncode == 0
    assert "usage: cornac" in proc.stdout and "--task-file" in proc.stdout


def test_python_m_cornac_without_a_task_exits_one():
    proc = _run_module()
    assert proc.returncode == 1
    assert "no task" in proc.stderr


def test_console_script_points_at_main():
    # `cornac = "cornac.cli:main"` is what makes the installed `cornac` command run main().
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    assert 'cornac = "cornac.cli:main"' in pyproject
