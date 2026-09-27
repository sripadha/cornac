# Observations — cornac lab notebook

Dated findings from running the harness against real local models. Numbers come from
the result files under `benchmark/spike/`; every claim here can be re-run. This is the
raw material for the write-up: what we measured, what surprised us, and what it taught
us about building a harness.

---

## 2026-09-22 — Choosing the benchmark model (three spike rounds)

**Question.** Which small local model should the coding benchmark use on an RTX 2060
(6 GB)? Decide on evidence, then freeze.

**Setup.** Three offline coding tasks (fix a failing test; one bug hides in a helper
module; one is a mutable-default-argument bug), each with a pristine fixture copied per
run and a cheat-resistant grader (real fix required, tests untouched, behavior checked
independently). Same harness, tools, prompt and context window for every model.
Round one used temperature 0; rounds two and 2b used 0.3 with seeds 42/43/44, because at
temperature 0 the three "repeats" were byte-identical runs.

**Round one** (`results/`): qwen2.5:7b 4/9, qwen3:8b 7/9, qwen3:4b 0/9 — but the last
is void: Ollama's `qwen3:4b` tag is the *thinking* variant, whose template cannot switch
thinking off, so it rambled instead of calling tools. Both 7–8B models were 1/3 on the
coding task. Not good enough to freeze.

**Round two** (`results_round2/`), coding only, 5 candidates:

| model | coding | s/run | GPU |
|---|---|---|---|
| Unsloth Qwen3.5-4B Q8_0 | 9/9 | 98 | 55% |
| Ollama `qwen3.5:4b` (Q4_K_M) | 7/9 | 25 | 100% |
| `qwen3:4b-instruct-2507` | 4/9 | 38 | 100% |
| `qwen3:8b` | 1/9 | 61 | 64% |
| Unsloth Qwen2.5-Coder-7B | 0/9 | — | — |

The Coder model was disqualified by a tool-call probe: it wrapped its call in `<xml>`
tags, so Ollama returned plain text and the harness saw no tool call at all.

**Round 2b** (`results_round2b/`). Ollama's `qwen3.5:4b` tag bakes `presence_penalty 1.5`
into its Modelfile. That setting discourages re-emitting text already in the context —
which is exactly what rewriting a file you just read consists of — and both of its
failures were whole-file writes that dropped the untouched functions. With the penalty
forced to 0: **9/9 at 20.5 s/run, 100% on the GPU.** The Unsloth Q6_K build also went
9/9 but at 51 s/run and 68% GPU.

**Decision.** Frozen on `qwen3.5:4b` with `presence_penalty 0`, `num_predict 2048`,
thinking off, sent by `OllamaProvider` on every request. Three builds of the same
weights tied at 9/9; the Ollama one is 2.5–5× faster and the only one that fits the card.

**Caveat.** 9 runs per model cannot separate 9/9 from "about 70%". The 270-run benchmark
is the real test. And 9/9 with tools means the benchmark tasks must be harder than the
spike tasks, or the upper rungs of the ladder will saturate.

---

## 2026-09-22 — The harness uplift (Week 4b-A acceptance)

**Question.** Round two's transcripts showed most coding failures were the harness's
fault, not the model's. If we fix the harness and change nothing else, do the same
models pass more?

**What changed.** Only cornac: an `edit_file` tool (replace one exact snippet instead
of rewriting the whole file); a syntax gate that refuses to save a broken `.py`;
`write_file` reporting "overwrote, 404 → 179 chars, removed def clamp, def mean"; a
bounded "continue" nudge when the model announces a step and stops; a note when the
exact same failed call is repeated; the workspace path in the system prompt.

**Result** (`results_round3/`), same tasks, seeds, temperature, prompts:

| model | old harness | new harness |
|---|---|---|
| `qwen3:4b-instruct-2507` | 4/9 | **9/9** |
| `qwen3:8b` | 1/9 | **8/9** |
| `qwen3.5:4b` (frozen) | 9/9 | 9/9 |
| total | 12/27 | **26/27** |

**Which pieces did the work.**
- `edit_file`: used in 25 of 27 runs in place of a whole-file write. Round two had 13
  whole-file writes that left a broken file; round three had zero broken files.
- The nudge: `qwen3:8b`'s "announced the fix, then stopped" count went 8 → 0. Five
  nudges were sent across two models; all five were followed by the described tool call,
  and all five runs passed.
- The syntax gate and the removed-functions report never fired: `edit_file` removed the
  cause upstream. Safety nets, not uplift.
- The repeated-call note fired 8 times and changed behavior 0 times. An 8B model re-sent
  the same failing edit six times while the note counted up. **Advisory text does not
  stop a determined loop.**
- The workspace hint removed the `cd /workspace` guessing seen in round two.

**The one remaining failure** is instructive: `qwen3:8b` found the correct fix but sent
it indented four spaces too far. `edit_file` diagnosed "matches except for whitespace"
and then refused to act on its own diagnosis. Harness gap; fix pending (dedent fallback).

---

## Why the 8B model lost to two 4B models

`qwen3:8b` went 1/9 with the old harness while newer 4B models did better, and 8/9 once
the harness improved. Size was not the axis being measured.

1. **It failed at acting, not at thinking.** In 8 of 9 round-two runs it diagnosed the
   bug correctly, wrote "Let's fix it and run the tests again", and ended its turn. It
   behaved like a chat assistant explaining a plan, not an agent executing one. A single
   harness sentence — "you described a step; do it" — fixed that 5 times out of 5. Its
   knowledge was fine; its follow-through was the problem.
2. **Newer training beats more parameters.** `qwen3:8b` is the April-2025 generation.
   `qwen3.5:4b` is a later generation trained heavily on tool use; even
   `qwen3:4b-instruct-2507`, the same family re-trained in July 2025, beat it 4/9 to 1/9
   on the old harness. A 2026 small model knows things about being an agent that a 2025
   larger model was never taught.
3. **Precision, not intelligence, decides coding tasks.** One wrong quote or indent
   fails the tests. Even in round three the 8B had a 20% tool-error rate versus 0% for
   both 4B models: it guessed `src/app.py` before listing the directory in 7 of 9 runs,
   and its one failure was a correct fix sent with the wrong indentation, re-sent six
   times. Size does not buy carefulness.

Not the reasons: the GPU spill (36% on CPU made it slow, but no run timed out) and the
quantization (all three were 4-bit).

**The lesson.** With the old harness the 8B looked like a bad coder; with the new one it
looked competent. Its capability never changed — what changed was whether the
scaffolding compensated for its habits. Bigger models fail for the same reasons small
ones do; they just fail more expensively.

---

## 2026-09-23 — Sub-agents: does a 4B model delegate?

**Question.** `spawn_agent` is a tool like any other, so whether the parent delegates is
the model's call — does the frozen `qwen3.5:4b` make it?

**Setup.** `examples/subagent_demo.py` (an eight-file blog package, `slugify` with a
leading/trailing-space bug, three callers plus a test) under the frozen options —
temperature 0, seed 42, `num_ctx` 8192, `presence_penalty` 0, think off — so each probe is
one deterministic sample, not a rate. Probe A is the demo as written, whose task says "use
spawn_agent" for the search; probe B (a scratch script, not in the repo) is the same
workspace, policy and tools with a neutral system prompt and that instruction removed.

**Result.**

| probe | delegated | steps | children | wall | tokens |
|---|---|---|---|---|---|
| A: task says "use spawn_agent" | yes, first turn | 6 | 1 (2 steps, 2.3k tokens) | 35.8 s | 17.8k |
| B: no instruction | no | 8 | 0 | 19.1 s | 21.7k |

Told to delegate, the 4B model delegated on its first turn and did it properly: a
self-contained task, a persona, exactly the read-only subset `["list_dir", "read_file",
"grep"]` it was asked for, and the child's four file names came back behind the
`Sub-agent reply:` header and were copied into the final answer as data, while the parent
read `utils.py` itself, made a one-line `edit_file` and confirmed with pytest. Left to
itself, it never touched `spawn_agent` — one `grep` call did the search — even though the
tool's description names "finding every place something is used" as its purpose; both
runs fixed the file and passed the tests. On an eight-file workspace not delegating is
the right call: one grep result is smaller than a child's whole loop, and the delegated
run took nearly twice the wall time for the same fix. Side note: in both runs the model
tried an ad-hoc `run_python` check, was denied (default ASK, no approver) and adapted; in
B it alternated two denied calls twice each before the repeated-call note fired and it
fell back to the allowed `python -m pytest -q` — n=1, but this time the note preceded a
change of approach.

**Lesson.** Context isolation only pays when there is context to isolate: the mechanism
works end to end on the frozen model, but whether delegation *helps* is a benchmark
question, and it needs tasks where the search would actually crowd the parent's window
(multi-file, longer files — the open item below).

---

## Harness design lessons collected so far

- **Errors are observations, not crashes.** A tool failure or a permission denial goes
  back to the model as a result it can read; the model adapts (pandas → csv; denied
  `run_python` → compute by hand).
- **A tool must never answer confidently when it did not look.** `grep` on a file path
  used to say "no matches" (it only walked directories). "I found nothing" and "I did
  not check" are different answers.
- **Gates fail closed.** A veto hook that crashes counts as a veto; an approver that
  returns anything but ALLOW is a refusal. A wrong refusal costs one round trip; a wrong
  execution cannot be undone.
- **Blocklists may be sloppy; allowlists must be precise.** `"ls" in "false"` is true.
  Allow = word-boundary prefix on the one argument the tool executes, no shell
  metacharacters; deny = substring over every argument.
- **Editing beats rewriting.** Whole-file writes were the single largest failure cause
  across every model. A one-line fix must stay a one-line fix.
- **Tell the model what it just did.** "Wrote 179 chars" hid the deletion of two
  functions; "overwrote 404 → 179 chars, removed def clamp, def mean" does not.
- **A bounded nudge converts narration into action.** One nudge, never more; the model
  sees exactly what happened as a user message (append-only context).
- **Advisory notes do not break loops.** After the second identical failed call the
  harness should change what it does, not what it says.
- **Sampling defaults are part of the model.** A baked-in `presence_penalty 1.5` turned
  a 9/9 model into a 7/9 one. Record the exact options on every run.
- **Temperature 0 is not three runs.** Repeats need a small temperature or they measure
  nothing.
- **Verify the tool channel before trusting a score.** A model that emits tool calls in
  the wrong wrapper looks like it "never uses tools"; probe first.

## Open items

- ~~`edit_file`: dedent fallback for whitespace-only mismatches; refuse `old == new`.~~
  Done 2026-09-22 (cleanup pass): a snippet that matches after adjusting indentation is
  applied and says so; an identical old/new is refused.
- Repeated identical failed calls: stop executing after the second and return the
  exact lines to copy instead. (Still open: the note is advisory only.)
- ~~Non-UTF-8 and CRLF files in `edit_file`; compile-based syntax gate; nudge regex
  false positives on "Let me summarize".~~ Done 2026-09-22 (cleanup pass).
- Benchmark tasks must be harder than the spike's (multi-file, longer files) so the
  ladder's top rungs do not saturate at 9/9.
- The "[cornac]" marker in tool results is spoofable by file contents (prompt-injection
  surface); documented, not solved.
- Sub-agents: two runs at temperature 0 say nothing about a delegation *rate*; measure
  on a workspace big enough that the search would crowd the parent, and compare the
  parent's own context size with and without the child.
