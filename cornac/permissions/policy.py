"""The permission policy — decides whether a tool call may run.

Week 2 showed the gap: the Workspace fence secures file tools, but run_bash/run_python
run arbitrary code that no path-fence can contain. The honest control is to decide,
*before* running, whether a given call is allowed — and to ask a human when unsure.

This module is pure decision logic (no I/O, no prompting). Given a ToolCall it returns
one of three Decisions:

    ALLOW  — run it, no questions
    DENY   — refuse; the model gets an error result and can adapt
    ASK    — undecided here; the caller (the agent) asks a human via prompt.py

Rules come from a small config (a dict, or YAML via Policy.from_yaml). Each tool maps to
either a plain decision ("allow"/"deny"/"ask") or, for command-bearing tools, a dict with
allow/deny substring lists plus a default:

    default: ask                       # for any tool not listed below
    tools:
      read_file: allow
      write_file: ask
      run_bash:
        deny:  ["rm -rf", "sudo", ":(){"]   # checked first — deny wins
        allow: ["pytest", "ls", "git status"]
        default: ask                         # neither matched -> ask a human
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from cornac.core.messages import ToolCall


class Decision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


# Sensible defaults matching the plan's tool table: read-only tools auto-allow,
# code-running and file-mutating tools ask first.
DEFAULT_RULES: dict = {
    "default": "ask",  # anything not listed -> ask
    "tools": {
        "read_file": "allow",
        "list_dir": "allow",
        "grep": "allow",
        "get_current_time": "allow",
        "web_search": "allow",
        "web_fetch": "allow",
        "write_file": "ask",
        "run_bash": {
            "deny": ["rm -rf", "sudo", "mkfs", ":(){", "dd if="],
            "default": "ask",
        },
        "run_python": "ask",
        "spawn_agent": "allow",
    },
}


class Policy:
    def __init__(self, rules: dict | None = None):
        rules = rules if rules is not None else DEFAULT_RULES
        self._default = Decision(rules.get("default", "ask"))
        self._tools: dict = rules.get("tools", {})
        # Session overrides set by "always"/"never" answers at the prompt. These take
        # precedence over the configured rules for the rest of the run.
        self._session: dict[str, Decision] = {}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Policy":
        import yaml

        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls(data)

    def remember(self, tool_name: str, decision: Decision) -> None:
        """Record an 'always allow'/'never' choice for the rest of this session."""
        self._session[tool_name] = decision

    def check(self, call: ToolCall) -> Decision:
        """Decide what to do with this tool call: ALLOW, DENY, or ASK."""
        # 1. A session override (from a previous "always"/"never") wins outright.
        if call.name in self._session:
            return self._session[call.name]

        # 2. No rule for this tool -> the global default.
        if call.name not in self._tools:
            return self._default

        rule = self._tools[call.name]

        # 3. Simple form: the rule is just "allow" / "deny" / "ask".
        if isinstance(rule, str):
            return Decision(rule)

        # 4. Rich form: a dict with deny/allow substring lists + a default. Match the
        #    lists against the call's argument text (e.g. the bash command).
        text = self._argument_text(call)
        for pattern in rule.get("deny", []):      # deny is checked first — it wins
            if pattern in text:
                return Decision.DENY
        for pattern in rule.get("allow", []):
            if pattern in text:
                return Decision.ALLOW
        return Decision(rule.get("default", "ask"))

    @staticmethod
    def _argument_text(call: ToolCall) -> str:
        """All string argument values joined — what allow/deny patterns match against.

        For run_bash this is the command; for run_python the code; in general, every
        string the model passed, so a deny pattern can't be hidden in an odd argument.
        """
        return " ".join(str(v) for v in call.arguments.values())
