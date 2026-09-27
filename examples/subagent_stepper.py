"""Sub-agents, one moment at a time — press Enter to advance.

The same task as subagent_demo.py (find every caller of a buggy helper, then fix it),
on the real model, but the run PAUSES at every moment that matters and prints, in
plain words, the one thing that is happening right then. Nothing is faked: the pauses
are hooks on the real loop, and the numbers you see are the real numbers.

    python examples/subagent_stepper.py            # local model, step by step
    python examples/subagent_stepper.py --auto     # same, no pauses (for a quick look)

If you take away one idea: spawn_agent is a tool whose job is to build a SECOND
agent — fresh memory, fewer tools, the same rules — run it, and hand back only its
final answer.
"""

import sys
import tempfile
import textwrap
from pathlib import Path

from cornac import Agent, HookBus, Policy, ToolRegistry
from cornac.tools.builtin import default_tools
from cornac.tools.workspace import Workspace

# Reuse the demo's workspace files and task text so this stepper never drifts from it.
from examples.subagent_demo import FILES, TASK, build_provider

AUTO = "--auto" in sys.argv
WIDTH = 78


# --- tiny display helpers ----------------------------------------------------------------

def box(title: str, body: str) -> None:
    print("\n" + "─" * WIDTH)
    print(f"● {title}")
    print("─" * WIDTH)
    print(textwrap.indent(textwrap.fill(body, WIDTH - 4), "  "))


def pause(prompt: str = "Enter to continue") -> None:
    if AUTO:
        return
    try:
        input(f"\n   [{prompt}] ")
    except EOFError:
        pass


def short(text: str, n: int = 110) -> str:
    text = text.replace("\n", " ⏎ ")
    return text if len(text) <= n else text[:n] + "…"


class Narrator:
    """Hooks on the PARENT bus that stop and explain each moment.

    The child's tool calls also arrive here (post_tool_use is forwarded from the
    child's bus), so we can tell the story of the whole tree from one place.
    """

    def __init__(self) -> None:
        self.depth = 0          # 0 = parent is acting, 1 = a child is acting
        self.parent_calls = 0

    # -- moment: the model answered --------------------------------------------------
    def on_assistant(self, msg) -> None:
        if msg.tool_calls:
            names = ", ".join(c.name for c in msg.tool_calls)
            box(f"The parent model replied — it wants to use: {names}",
                "The model cannot run anything itself. It has only ASKED for a tool. Now the "
                "loop will decide whether to run it (hook veto → policy → approver), then run "
                "it and feed the result back.")
            for c in msg.tool_calls:
                if c.name == "spawn_agent":
                    self.explain_spawn_request(c)
        else:
            box("The parent model replied with text and NO tool call — so it is done",
                short(msg.text or "", 300))

    def explain_spawn_request(self, call) -> None:
        a = call.arguments
        tools = a.get("tools") or "(all of the parent's tools)"
        box("Look at HOW the model asked to delegate — three arguments",
            "task: the complete instructions, because the child will start with an EMPTY "
            "memory and knows nothing about this conversation.")
        print(textwrap.indent(textwrap.fill(f"task = {short(a.get('task', ''), 260)}", WIDTH - 6), "    "))
        print(textwrap.indent(textwrap.fill(f"system_prompt = {short(a.get('system_prompt') or '(default persona)', 160)}", WIDTH - 6), "    "))
        print(textwrap.indent(f"tools = {tools}", "    "))
        print("\n  tools: only READ-ONLY ones. The child gets a subset of the parent's tools,")
        print("  never more — and never spawn_agent itself.")
        pause("Enter: the loop will now build the child")

    # -- moment: a child is about to run --------------------------------------------
    def on_spawn(self, task: str, depth: int) -> None:
        self.depth = depth
        box(f"A CHILD agent is being built (depth {depth})",
            "spawn_agent's run() just did this:  child = Agent(provider=parent.provider, "
            "registry=<the 3 read-only tools>, policy=parent.policy, approver=parent.approver, "
            "hooks=<its own bus, wired back to this one>, max_steps=10).  Same brain, same "
            "rulebook, same human — the SAME objects, not copies — so the child can never do "
            "something the parent could not. Its conversation starts empty: just its system "
            "prompt and the task.")
        pause("Enter: watch the child work (its tool calls appear indented)")

    # -- moment: any tool ran (parent's or child's) ----------------------------------
    def on_post_tool(self, call, result) -> None:
        who = "CHILD" if self.depth > 0 else "PARENT"
        indent = "      " if self.depth > 0 else "  "
        flag = " [ERROR]" if result.is_error else ""
        print(f"{indent}↳ {who} ran {call.name}{flag} → {short(result.content, 120)!r}")
        if self.depth > 0:
            print(f"{indent}  (this line reached the parent's bus because the child forwards its "
                  f"tool-call events — a veto hook here would have blocked it)")
        elif call.name == "spawn_agent":
            self.explain_reply(result)

    def explain_reply(self, result) -> None:
        box("What the parent got back from the child",
            "Only the child's FINAL text, behind a 'Sub-agent reply:' header, trimmed if long. "
            "None of the child's reading, grepping, or dead ends — that scratch work stayed "
            "in the child's memory and was thrown away with it. That is the whole point: the "
            "parent's context holds one short answer instead of a pile of file contents.")
        print(textwrap.indent(result.content, "    │ "))
        pause("Enter: the parent continues with a clear head")

    # -- moment: the child finished -------------------------------------------------
    def on_spawn_done(self, result, depth: int) -> None:
        self.depth = 0
        if result is None:
            box("The child CRASHED", "The parent gets an error result and its loop carries on.")
            return
        box(f"The child finished (depth {depth})",
            f"stop_reason={result.stop_reason}, steps={result.steps}, "
            f"tokens={result.usage.total_tokens}. Its RunResult tells the truth: 'done' means "
            f"there is a real answer; 'max_steps' would mean it gave up, and the parent would "
            f"get an ERROR, never an empty string it might mistake for 'nothing found'. "
            f"The parent now ABSORBS the child's {result.usage.total_tokens} tokens into its "
            f"own bill (absorb_child), because the benchmark must count what delegation cost.")
        pause("Enter: see the reply the parent receives")

    # -- moment: a permission decision (parent or child) ------------------------------
    def on_decision(self, call, decision) -> None:
        if decision.value != "allow":
            who = "child" if self.depth > 0 else "parent"
            print(f"   ⚠ policy said {decision.value.upper()} for the {who}'s {call.name} "
                  f"(same rulebook object for both — that is why)")


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="cornac_step_"))
    for name, content in FILES.items():
        (tmp / name).write_text(content)
    workspace = Workspace(tmp)

    box("THE SETUP",
        f"A blog package of {len(FILES)} files in {tmp}. utils.py has a helper slugify() with "
        "a bug (it does not strip spaces), three files call it, and one test fails. The task: "
        "find every caller (a noisy search) and fix the bug (a precise edit). We will watch "
        "the parent hand the search to a CHILD agent and keep the edit for itself.")
    pause("Enter to start the real model")

    n = Narrator()
    bus = HookBus()
    bus.on("on_assistant_message", n.on_assistant)
    bus.on("on_spawn", n.on_spawn)
    bus.on("on_spawn_done", n.on_spawn_done)
    bus.on("post_tool_use", n.on_post_tool)
    bus.on("on_permission_decision", n.on_decision)

    policy = Policy({"default": "ask", "tools": {
        "list_dir": "allow", "read_file": "allow", "grep": "allow",
        "edit_file": "allow", "spawn_agent": "allow",
        "run_bash": {"allow": ["python -m pytest", "pytest"],
                     "deny": ["rm -rf", "sudo"], "default": "ask"},
    }})

    agent = Agent(
        provider=build_provider("ollama"),
        registry=ToolRegistry(default_tools(workspace)),   # ends with SpawnAgent(); binding happens here
        system_prompt=("You are a careful software engineer working in a sandboxed "
                       "workspace. Delegate exploration to sub-agents when the task "
                       "says so; make edits yourself.\n\n" + workspace.describe()),
        max_steps=12,
        policy=policy,
        hooks=bus,
    )

    box("THE PLUMBING YOU DON'T SEE",
        "Agent(...) was just built. Its __init__ walked the tools and called "
        "tool.bind_parent(agent) on spawn_agent — that is how the tool knows whose brain, "
        "rules and hooks to hand to a child later. Every agent.run() also calls "
        "tool.reset_for_run(), so the 'at most 5 children per run' counter starts fresh.")
    pause("Enter: agent.run(task) — the ordinary loop begins")

    result = agent.run(TASK)

    box("THE RUN IS OVER",
        f"stop_reason={result.stop_reason}  steps={result.steps} (parent only)  "
        f"children={result.children}  nudges={result.nudges}  "
        f"tokens={result.usage.total_tokens} (child's included)  "
        f"duration={result.duration:.1f}s")
    print("\n  FINAL ANSWER:")
    print(textwrap.indent(result.text or "(none)", "    "))
    fixed = "strip()" in (tmp / "utils.py").read_text()
    print(f"\n  [check on disk] utils.py {'now strips whitespace ✔' if fixed else 'was NOT fixed ✘'}")
    print("\n  Parent's conversation length:", len(agent.messages), "messages —",
          "the child's exploration is not in it.")


if __name__ == "__main__":
    main()
