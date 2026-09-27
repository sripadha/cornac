"""Week 4b demo — delegate the noisy part of a task to a SUB-AGENT.

The workspace is a small blog package: eight Python files, three of which call a
helper `slugify` from utils.py, and the helper has a bug. The task has two halves
with very different shapes:

  1. find every caller of slugify — a search that means opening and reading files,
     whose contents would otherwise sit in the parent's context for the rest of the
     run and crowd out what it was doing;
  2. fix the one-line bug and prove it with the tests — a precise edit that needs a
     clear head.

So the parent is asked to hand the search to a sub-agent with read-only tools. The
child does the reading in its own context and returns one line: the list of callers.
The parent's transcript holds that line and nothing else from the search, and its
RunResult still reports what the whole thing cost, child included.

Two hooks narrate the delegation: on_spawn and on_spawn_done fire on the PARENT's bus
with the depth of the spawn, and on_spawn_done carries the child's RunResult, so you
can see its steps and tokens without ever seeing its transcript. The child's TOOL
CALLS do reach this bus — post_tool_use is forwarded, so the "↳" lines between a
spawn and its done are the child's, and a veto hook registered here would bind the
child too — but its assistant turns and its on_stop do not: a sub-agent's dialogue
is its own business.

    python examples/subagent_demo.py ollama
    python examples/subagent_demo.py anthropic   # needs ANTHROPIC_API_KEY

Whether the model actually delegates is the model's call — spawn_agent is a tool like
any other — but the task tells it to, and the frozen benchmark model does.
"""

import sys
import tempfile
from pathlib import Path

from cornac import Agent, HookBus, Policy, ToolRegistry
from cornac.tools.builtin import default_tools
from cornac.tools.workspace import Workspace

# --- the workspace: a small blog package with one buggy helper ---------------------------

FILES = {
    "utils.py": '''\
"""Small helpers shared across the blog package."""


def slugify(title: str) -> str:
    """Turn a post title into a URL slug: lower-case, words joined by dashes.

    slugify("Hello World") -> "hello-world"
    """
    return title.lower().replace(" ", "-")


def word_count(text: str) -> int:
    return len(text.split())
''',
    "models.py": '''\
"""The one data type: a blog post."""

from dataclasses import dataclass


@dataclass
class Post:
    title: str
    body: str
    author: str = "anon"
''',
    "config.py": '''\
"""Site-wide settings."""

SITE_NAME = "Tiny Blog"
POSTS_PER_PAGE = 10
OUTPUT_DIR = "out"
''',
    "posts.py": '''\
"""Post URLs: /posts/<slug>."""

from models import Post
from utils import slugify


def post_url(post: Post) -> str:
    return f"/posts/{slugify(post.title)}"
''',
    "export.py": '''\
"""Write each post to <OUTPUT_DIR>/<slug>.html."""

from config import OUTPUT_DIR
from models import Post
from render import render_post
from utils import slugify


def export_path(post: Post) -> str:
    return f"{OUTPUT_DIR}/{slugify(post.title)}.html"


def export(post: Post) -> tuple[str, str]:
    return export_path(post), render_post(post)
''',
    "feeds.py": '''\
"""Stable ids for the RSS feed."""

from models import Post
from utils import slugify


def feed_id(post: Post) -> str:
    return "tag:tinyblog," + slugify(post.title)
''',
    "render.py": '''\
"""Render a post as (very plain) HTML."""

from models import Post
from utils import word_count


def render_post(post: Post) -> str:
    return (
        f"<h1>{post.title}</h1>\\n"
        f"<p class=meta>{post.author} · {word_count(post.body)} words</p>\\n"
        f"<div>{post.body}</div>\\n"
    )
''',
    "main.py": '''\
"""Build the site: one page per post."""

from export import export
from models import Post
from posts import post_url

POSTS = [
    Post("Hello World", "The first post."),
    Post(" Trailing Space ", "A title someone typed carelessly."),
]


def build() -> list[str]:
    urls = []
    for post in POSTS:
        path, html = export(post)
        urls.append(post_url(post))
    return urls


if __name__ == "__main__":
    print("\\n".join(build()))
''',
    "test_utils.py": '''\
from utils import slugify


def test_plain_title():
    assert slugify("Hello World") == "hello-world"


def test_surrounding_whitespace_is_ignored():
    assert slugify(" Trailing Space ") == "trailing-space"
''',
}

TASK = (
    "The helper `slugify` in utils.py has a bug: a title with leading or trailing "
    "spaces gets a dash at the start or end (slugify(' Trailing Space ') returns "
    "'-trailing-space-' instead of 'trailing-space').\n\n"
    "Do this in two steps.\n"
    "1. Use spawn_agent to find every file that calls slugify. Give the sub-agent only "
    "the read-only tools (list_dir, read_file, grep) and tell it to reply with just the "
    "file names, one per line.\n"
    "2. Fix the bug in utils.py yourself with edit_file, then run "
    "`python -m pytest -q` to confirm the tests pass.\n\n"
    "Reply with the list of callers and a one-line description of the fix."
)


def build_provider(which: str):
    if which == "anthropic":
        from cornac.providers.anthropic import AnthropicProvider

        return AnthropicProvider()
    from cornac.providers.ollama import OllamaProvider

    return OllamaProvider()


def install_hooks(bus: HookBus) -> None:
    """Narrate the parent's tool calls and, above all, the delegation."""

    def on_assistant(m):
        if m.tool_calls:
            for c in m.tool_calls:
                args = {k: (v[:80] + "..." if isinstance(v, str) and len(v) > 80 else v)
                        for k, v in c.arguments.items()}
                print(f"[parent] calls {c.name}({args})")
        elif m.text:
            print(f"[parent] says {m.text[:160]!r}")
    bus.on("on_assistant_message", on_assistant)

    # on_spawn(task, depth): a child is about to run. depth is 1 for a direct child of
    # this agent, 2 for a grandchild — those are forwarded up, so the tree is visible
    # from here.
    bus.on("on_spawn", lambda task, depth: print(f"   [spawn depth={depth}] task: {task[:120]!r}"))

    # on_spawn_done(result, depth): the child returned. `result` is its RunResult —
    # or None if the child crashed (a provider error, say).
    def on_spawn_done(result, depth):
        if result is None:
            print(f"   [spawn depth={depth}] child failed before returning a result")
            return
        print(f"   [spawn depth={depth}] done: stop_reason={result.stop_reason} "
              f"steps={result.steps} tokens={result.usage.total_tokens} "
              f"reply={len(result.text or '')} chars")
    bus.on("on_spawn_done", on_spawn_done)

    def on_post_tool(call, result):
        body = result.content.replace("\n", " ⏎ ")
        body = (body[:120] + "...") if len(body) > 120 else body
        flag = " [ERROR]" if result.is_error else ""
        print(f"   ↳ {call.name} ->{flag} {body!r}")
    bus.on("post_tool_use", on_post_tool)

    bus.on("on_stop", lambda text: print("\n[stop] final answer ready" if text is not None
                                         else "\n[stop] hit max_steps"))


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "ollama"

    tmp = Path(tempfile.mkdtemp(prefix="cornac_subagent_"))
    for name, content in FILES.items():
        (tmp / name).write_text(content)
    workspace = Workspace(tmp)

    bus = HookBus()
    install_hooks(bus)

    # Non-interactive: the tools this task needs are pre-allowed, pytest included.
    # Everything else (write_file, run_python, a non-pytest command) would ASK, and
    # with no approver an ASK is denied — in the child exactly as in the parent.
    policy = Policy({"default": "ask", "tools": {
        "list_dir": "allow", "read_file": "allow", "grep": "allow",
        "edit_file": "allow", "spawn_agent": "allow",
        "run_bash": {"allow": ["python -m pytest", "pytest"],
                     "deny": ["rm -rf", "sudo"], "default": "ask"},
    }})

    agent = Agent(
        provider=build_provider(which),
        # default_tools() ends with SpawnAgent(); building the Agent binds it.
        registry=ToolRegistry(default_tools(workspace)),
        system_prompt=("You are a careful software engineer working in a sandboxed "
                       "workspace. Delegate exploration to sub-agents when the task "
                       "says so; make edits yourself.\n\n" + workspace.describe()),
        max_steps=12,
        policy=policy,
        hooks=bus,
    )

    print(f"=== subagent_demo | provider: {agent.provider.name} | workspace: {tmp} ===\n")
    result = agent.run(TASK)

    print(f"\n=== FINAL ANSWER ===\n{result.text}")
    print(f"\n[run] stop_reason={result.stop_reason} steps={result.steps} "
          f"children={result.children} nudges={result.nudges} "
          f"duration={result.duration:.1f}s tokens={result.usage.total_tokens} "
          f"(in {result.usage.input_tokens} / out {result.usage.output_tokens}; "
          f"children's tokens included)")

    # What actually landed on disk, so the demo is checkable without trusting the model.
    fixed = "strip()" in (tmp / "utils.py").read_text()
    print(f"[check] utils.py {'now strips whitespace' if fixed else 'was NOT fixed'}")


if __name__ == "__main__":
    main()
