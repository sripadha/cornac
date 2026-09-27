"""The command line: `cornac "task"` — one agent run from a terminal.

Until now, running an agent meant writing a script (see examples/). This module is
that script, written once: it parses the arguments, builds an Agent from them, runs
the task, prints what came back and turns the outcome into an exit code. It is meant
to be read in one sitting, so the three jobs are three functions and nothing here is
clever:

    parse_args(argv)            -> the options, with the task text resolved
    build_agent(args, provider) -> a ready Agent (provider injectable for tests)
    print_result(result, quiet) -> the answer or the honest "did not finish", plus a footer
    main(argv)                  -> glue, error handling, exit code

Exit codes, because a shell script or a CI job reads those and not the prose:

    0   the model finished ("done")
    2   the loop gave up (max_steps or stuck) — partial work was printed, but it is
        not an answer, and the exit code refuses to pretend otherwise
    1   a usage or provider error (bad flags, unreadable task file, daemon down,
        missing API key); the message goes to stderr

Why the unfinished case prints anything at all: the 2026 harness comparison flagged
"throw the child's work away at max_steps" as the one place other harnesses beat
cornac. So an unfinished run now carries a `digest` (what the harness saw happen,
no model involved) and a `summary` (the model's own four-line account, from one
extra call with tools switched off). Both are printed here — under a header that
says the run did not finish, so nobody skims the summary and takes it for a result.

Permissions on the command line: the built-in policy allows only the read-only
tools (read_file, list_dir, grep, get_current_time) and ASKS about everything else,
with cli_ask as the approver — so a first-time user sees every write and every
command before it runs, and can answer "always" to stop being asked. run_bash also
carries the library's deny list ("rm -rf", "sudo", ...), and that is what makes
"always" safe to offer: an "always" only answers the QUESTION, and a configured deny
still refuses (see policy.py, "Session overrides"). Without the list, one "a" on
`pytest -q` would have let `sudo rm -rf ~` run unasked for the rest of the session —
on the surface a beginner is most likely to use. `--policy` loads a YAML file instead
(examples/cornac.policy.yaml is a starting point).

Runs as `python -m cornac ...` (see __main__.py) and as the `cornac` console script
(pyproject.toml, [project.scripts]); both call main() and exit with its return value.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

from cornac.core.agent import Agent
from cornac.core.result import RunResult
from cornac.permissions.policy import DEFAULT_RULES, Policy
from cornac.permissions.prompt import cli_ask
from cornac.prompts import build_system_prompt
from cornac.providers.base import Provider
from cornac.tools.builtin import default_tools
from cornac.tools.registry import ToolRegistry
from cornac.tools.workspace import Workspace

# The policy used when no --policy file is given. Read-only tools run without a
# question; everything that writes, runs code, reaches the network or spawns an agent
# asks the human first. Deliberately stricter than policy.DEFAULT_RULES (which
# auto-allows web_search/web_fetch/spawn_agent for the library's examples): on the
# command line the human IS there to answer, so the safe default costs one keystroke
# and the over-permissive one is not worth it.
#
# run_bash keeps DEFAULT_RULES' rich rule — the same deny list, still "ask" by
# default — rather than falling to the bare top-level "ask". The difference only
# shows after the user answers "always": a plain "ask" has no deny list for the
# session override to leave in force, so every later run_bash, `sudo rm -rf ~`
# included, would run without a question, while cli_ask was telling the user "its
# deny patterns still apply". Copied, not shared, so neither dict can edit the other.
CLI_POLICY: dict = {
    "default": "ask",
    "tools": {
        "read_file": "allow",
        "list_dir": "allow",
        "grep": "allow",
        "get_current_time": "allow",
        "run_bash": copy.deepcopy(DEFAULT_RULES["tools"]["run_bash"]),
    },
}

PROVIDERS = ("ollama", "anthropic", "openai")


class UsageError(Exception):
    """A command-line mistake (bad flag, no task, unreadable task file).

    argparse's own error() calls sys.exit(2), and 2 already means "the run did not
    finish" here. Raising instead lets main() report every usage problem the same
    way — message on stderr, exit code 1 — and keeps tests free of SystemExit.
    """


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        raise UsageError(f"{self.format_usage().rstrip()}\n{self.prog}: error: {message}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line. On return, `args.task` always holds the task text.

    The task comes either as the one positional argument or from --task-file (a
    multi-paragraph task is easier to keep in a file than to quote in a shell);
    exactly one of the two is required, and the file is read here so that the rest
    of the program never has to care which was used.
    """
    p = _Parser(
        prog="cornac",
        description="Run one cornac agent on a task, with tools confined to a workspace.",
    )
    p.add_argument("task", nargs="?", help="the task, as one quoted string")
    p.add_argument("--task-file", metavar="FILE", help="read the task from this file instead")
    p.add_argument("--provider", choices=PROVIDERS, default="ollama",
                   help="which model backend to talk to (default: ollama)")
    p.add_argument("--model", help="model name/tag; each provider has its own default")
    p.add_argument("--base-url", metavar="URL",
                   help="server URL for --provider openai (or the Ollama host)")
    p.add_argument("--workspace", metavar="DIR", default=".",
                   help="the directory the tools are confined to (default: cwd)")
    p.add_argument("--policy", metavar="FILE.yaml",
                   help="permission policy to load (default: read-only tools allowed, "
                        "everything else asks)")
    p.add_argument("--max-steps", type=int, default=20, metavar="N",
                   help="model calls before the loop gives up (default: 20)")
    p.add_argument("--transcript", metavar="PATH",
                   help="write every event of the run to this JSONL file")
    p.add_argument("--no-instructions", action="store_true",
                   help="do not read AGENTS.md / CLAUDE.md from the workspace")
    p.add_argument("--quiet", action="store_true", help="print the result only, no footer")
    args = p.parse_args(argv)

    if args.task and args.task_file:
        p.error("give the task as an argument OR with --task-file, not both")
    if args.task_file:
        try:
            args.task = Path(args.task_file).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            # UnicodeDecodeError is a ValueError, not an OSError: a task file that is
            # not UTF-8 must get the same one-line usage error as a missing one, not
            # a traceback.
            p.error(f"cannot read task file: {exc}")
    if not args.task or not args.task.strip():
        p.error("no task given (pass it as an argument or with --task-file)")
    if args.max_steps < 1:
        p.error("--max-steps must be at least 1")
    return args


def make_provider(args: argparse.Namespace) -> Provider:
    """Build the provider the flags ask for.

    Each import is local so that `cornac` starts with whatever is installed: the
    Anthropic SDK is only needed for --provider anthropic, and importing it (or
    reading its API key) for an Ollama run would be a pointless failure. Tests
    replace this function to inject a scripted provider.
    """
    if args.provider == "anthropic":
        from cornac.providers.anthropic import AnthropicProvider

        return AnthropicProvider(**({"model": args.model} if args.model else {}))
    if args.provider == "openai":
        from cornac.providers.openai_compat import OpenAICompatibleProvider

        kwargs = {"model": args.model or ""}
        if args.base_url:
            kwargs["base_url"] = args.base_url
        return OpenAICompatibleProvider(**kwargs)
    from cornac.providers.ollama import OllamaProvider

    kwargs = {}
    if args.model:
        kwargs["model"] = args.model
    if args.base_url:
        kwargs["host"] = args.base_url
    return OllamaProvider(**kwargs)


def build_agent(args: argparse.Namespace, provider: Provider | None = None) -> Agent:
    """Turn parsed arguments into a ready-to-run Agent.

    This is the same wiring every example does by hand — workspace, tools, policy,
    approver, system prompt — in one place. `provider` is injectable so a test can
    hand in a scripted one and exercise everything else for real: the fence, the
    policy, the prompt. The --transcript import is deferred to its branch so the
    CLI works even where the transcript module is absent or broken.
    """
    workspace = Workspace(args.workspace)
    policy = Policy.from_yaml(args.policy) if args.policy else Policy(CLI_POLICY)
    agent = Agent(
        provider=provider if provider is not None else make_provider(args),
        registry=ToolRegistry(default_tools(workspace)),
        system_prompt=build_system_prompt(workspace, instructions=not args.no_instructions),
        max_steps=args.max_steps,
        policy=policy,
        approver=cli_ask,
    )
    if args.transcript:
        from cornac.transcript import TranscriptWriter

        TranscriptWriter(args.transcript).attach(agent)
    return agent


def print_result(result: RunResult, quiet: bool = False, out=None) -> None:
    """Print the outcome: the answer, or the labelled partial work, then one footer line.

    A finished run prints its text. An unfinished one prints a header that says so,
    then the harness's digest and the model's own summary when they exist. The
    header always comes first and the summary is labelled "not an answer": a model
    that ran out of steps was often going in circles, and a model in that state
    writes summaries that sound like success. The footer is one grep-able line —
    `stop=done steps=3 tokens=1234 time=4.2s` — and --quiet drops it for callers
    that only want the text.
    """
    out = out or sys.stdout
    if result.stop_reason == "done":
        if result.text:
            print(result.text, file=out)
    else:
        print(f"Did not finish ({result.stop_reason}).", file=out)
        if result.digest:
            print(f"\nWhat happened (harness digest):\n{result.digest}", file=out)
        if result.summary:
            print(f"\nThe model's own account (partial work, not an answer):\n{result.summary}",
                  file=out)
    if not quiet:
        print(f"stop={result.stop_reason} steps={result.steps} "
              f"tokens={result.usage.total_tokens} time={result.duration:.1f}s", file=out)


def main(argv: list[str] | None = None) -> int:
    """Entry point: parse, build, run, print, and turn the outcome into an exit code.

    Errors are reported as one line on stderr, not a traceback: the person at the
    terminal wants to know that the Ollama daemon is down or the API key is missing,
    not where in httpx it was noticed. Anything the run raises is a provider or setup
    failure by construction — the loop itself turns tool failures into results for
    the model — so catching Exception here is honest, not lazy.
    """
    try:
        args = parse_args(argv)
    except UsageError as exc:
        print(exc, file=sys.stderr)
        return 1
    try:
        agent = build_agent(args)
        result = agent.run(args.task)
    except KeyboardInterrupt:
        print("\ncornac: interrupted", file=sys.stderr)
        return 130  # the shell's convention for "killed by Ctrl-C"
    except Exception as exc:  # noqa: BLE001 — see the docstring
        print(f"cornac: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print_result(result, quiet=args.quiet)
    return 0 if result.stop_reason == "done" else 2
