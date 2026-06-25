"""Interactive CLI approval — what happens when the policy returns ASK.

When the policy can't decide, the agent pauses and asks a human right in the terminal:

    cornac wants to run a tool:
      run_bash(command='pytest tests/')
    Allow? [y]es / [n]o / [a]lways / nev[e]r:

  y  -> allow this once
  n  -> deny this once (model gets an error result, can adapt)
  a  -> allow, and remember: auto-allow this tool for the rest of the session
  e  -> deny, and remember: auto-deny this tool for the rest of the session

The "always/never" answers call policy.remember(...) so you're not re-asked every time.

This is deliberately a small, swappable function. A web UI, a Slack bot, or an automated
test could provide a different approver with the same signature — the agent only needs
"given a call, return a Decision".
"""

from __future__ import annotations

from cornac.core.messages import ToolCall
from cornac.permissions.policy import Decision, Policy


def _format_call(call: ToolCall) -> str:
    args = ", ".join(f"{k}={v!r}" for k, v in call.arguments.items())
    return f"{call.name}({args})"


def cli_ask(call: ToolCall, policy: Policy | None = None) -> Decision:
    """Ask the human on the terminal whether to run `call`. Returns ALLOW or DENY.

    If `policy` is given, an "always"/"never" answer is remembered on it so the same
    tool isn't re-asked for the rest of the session.
    """
    print("\ncornac wants to run a tool:")
    print(f"  {_format_call(call)}")

    while True:
        try:
            answer = input("Allow? [y]es / [n]o / [a]lways / nev[e]r: ").strip().lower()
        except EOFError:
            # No interactive input available (piped/closed stdin) -> safe default: deny.
            print("(no input — denying)")
            return Decision.DENY
        if answer in ("y", "yes"):
            return Decision.ALLOW
        if answer in ("n", "no", ""):
            return Decision.DENY
        if answer in ("a", "always"):
            if policy is not None:
                policy.remember(call.name, Decision.ALLOW)
            return Decision.ALLOW
        if answer in ("e", "never"):
            if policy is not None:
                policy.remember(call.name, Decision.DENY)
            return Decision.DENY
        print("  please answer y, n, a, or e")
