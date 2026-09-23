"""Tests for cli_ask — the terminal prompt behind an ASK decision.

cli_ask is the human approval gate and, in production, the only caller of
Policy.remember(). It reads answers with input(), so builtins.input is monkeypatched
with a scripted iterator and no terminal is needed. What is pinned here: the y/n/a/e
parsing, that "always"/"never" are remembered on the policy, that an invalid answer
re-prompts rather than guessing, and that a closed stdin (EOFError) means DENY —
the safe default when nobody can answer.
"""

import pytest

from cornac.core.messages import ToolCall
from cornac.permissions.policy import Decision, Policy
from cornac.permissions.prompt import cli_ask

CALL = ToolCall("1", "run_bash", {"command": "pytest -q"})


@pytest.fixture
def policy():
    return Policy({"tools": {"run_bash": "ask"}})


def answers(monkeypatch, *scripted: str) -> None:
    """Make input() hand back each scripted answer in turn."""
    it = iter(scripted)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(it))


@pytest.mark.parametrize("answer, expected", [
    ("y", Decision.ALLOW),
    ("yes", Decision.ALLOW),
    (" Y ", Decision.ALLOW),      # whitespace and case are forgiven
    ("n", Decision.DENY),
    ("no", Decision.DENY),
    ("", Decision.DENY),          # bare Enter: the safe answer, not the convenient one
])
def test_one_shot_answers(monkeypatch, policy, answer, expected):
    answers(monkeypatch, answer)
    assert cli_ask(CALL, policy) == expected
    assert policy.check(CALL) == Decision.ASK   # a one-shot answer is not remembered


def test_always_allows_and_is_remembered(monkeypatch, policy):
    answers(monkeypatch, "a")
    assert cli_ask(CALL, policy) == Decision.ALLOW
    assert policy.check(CALL) == Decision.ALLOW   # no longer ASK for this tool


def test_never_denies_and_is_remembered(monkeypatch, policy):
    answers(monkeypatch, "e")
    assert cli_ask(CALL, policy) == Decision.DENY
    assert policy.check(CALL) == Decision.DENY


def test_always_does_not_switch_off_the_deny_list(monkeypatch):
    # "always" only skips the question; a configured deny still refuses.
    policy = Policy({"tools": {"run_bash": {"deny": ["rm -rf"], "default": "ask"}}})
    answers(monkeypatch, "a")
    assert cli_ask(CALL, policy) == Decision.ALLOW
    assert policy.check(CALL) == Decision.ALLOW
    assert policy.check(ToolCall("2", "run_bash", {"command": "rm -rf /"})) == Decision.DENY


def test_always_and_never_work_without_a_policy_to_remember_on(monkeypatch):
    answers(monkeypatch, "a")
    assert cli_ask(CALL, None) == Decision.ALLOW
    answers(monkeypatch, "e")
    assert cli_ask(CALL, None) == Decision.DENY


def test_an_invalid_answer_re_prompts_instead_of_guessing(monkeypatch, capsys):
    answers(monkeypatch, "zzz", "y")
    assert cli_ask(CALL) == Decision.ALLOW
    assert "please answer y, n, a, or e" in capsys.readouterr().out


def test_closed_stdin_means_deny(monkeypatch, capsys):
    # Piped or closed stdin: input() raises EOFError. Nobody can approve, so deny.
    def no_terminal(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", no_terminal)
    assert cli_ask(CALL) == Decision.DENY
    assert "denying" in capsys.readouterr().out


def test_the_prompt_shows_the_call_on_stdout(monkeypatch, capsys):
    answers(monkeypatch, "n")
    cli_ask(CALL)
    out = capsys.readouterr().out
    assert "cornac wants to run a tool" in out
    assert "run_bash(command=\'pytest -q\')" in out
