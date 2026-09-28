# The capability ladder

One experiment, six levels. The **same frozen model** (`qwen3.5:4b` on Ollama, temperature
0.3, seeds 42 + run, num_ctx 8192, num_predict 2048, presence_penalty 0, think off, exactly
round three's settings) fixes the **same three failing-test tasks** (`benchmark/spike/tasks/`
swe, swe2, swe3: pytest fails, fix the source, do not touch the test) **k=3 times per level**.
Each level adds exactly one thing to the level below, so the gap between two rows is what
that one thing was worth. Fixed at every level: the task prompt from `task.md`, a fresh copy
of the fixture per run, grading by the task's `verify.py` (it reads the workspace, never the
prose), max_steps 12, headless (an ASK becomes a DENY), and one permission policy (allow the
file, python and spawn tools; run_bash minus `rm -rf`, `sudo`, `mkfs`, `dd if=`, `:(){`). The
policy is a safety property, not a capability, so it is the same on every rung.

| # | key | adds (exactly one thing) | the question it answers |
|---|---|---|---|
| 0 | `bare` | nothing: the prompt with the files pasted in, one model call, no system prompt | can the raw model do it alone? |
| 1 | `prompted` | a system prompt (persona + the file list); the user message is identical to level 0 | does telling it the situation help? |
| 2 | `one_tool` | one tool, `run_bash`; the files are no longer pasted, it can look | does being able to act help? |
| 3 | `tools` | the full read/write set: read_file, write_file, list_dir, grep, run_bash, run_python; guards off (the round-two harness) | do more tools help? |
| 4 | `harness` | edit_file, the syntax gate, the change report, the nudge, repeat note, hard stop, wrap-up, context clearing (Weeks 4b + 4c) | does the harness help? |
| 5 | `agents` | `spawn_agent` and one sentence allowing delegation; children inherit level 4's loop settings | does delegation help? |

Every text the model reads at a level names only the tools that level has: `run_bash`,
`run_python` and `write_file` carry round two's descriptions at levels 2-3 (no "use
edit_file" hint for a tool that is not there) and today's at 4-5, and the workspace
paragraph in the system prompt names only the run tools present. A model told about a
tool it does not have calls a tool that does not exist, and that penalty would land on
the wrong row.

`levels.py` is the single source of truth for what each level is (`LEVELS`, one `Level`
each); the runner and the six `examples/stage_*.py` scripts import it, they never redefine it.
Level 4 is the closest rung to round three of the model spike, where the tool half of the
harness took three models from 12/27 to 26/27 on these tasks (`docs/observations.md`,
`benchmark/spike/results_round3/`).

## Run it

```bash
source .venv/bin/activate                 # pytest must be installed in this interpreter: the tasks run it
                                          # (the runner also puts it first on PATH for run_bash)
ollama pull qwen3.5:4b
python benchmark/ladder/run_ladder.py     # all six levels x three tasks x k=3 (54 runs)
python benchmark/ladder/run_ladder.py --levels 0,2,5 --tasks swe --k 1     # a quick look
python benchmark/ladder/run_ladder.py --resume --out benchmark/ladder/results/<dir>  # continue
python benchmark/ladder/run_ladder.py --report-only benchmark/ladder/results/<dir>    # rebuild results.md
```

Flags: `--levels 0-5|0,2,5|3`, `--tasks swe,swe2,swe3`, `--k 3`, `--temperature 0.3`,
`--seed-base 42`, `--max-steps 12`, `--model qwen3.5:4b`, `--provider ollama|openai`,
`--base-url URL`, `--out DIR`, `--resume`. Runs go level-major (all of level 0, then level 1,
...), so an interrupted run still has complete rows for the levels it finished, and `--resume`
skips the (level, task, run) triples already recorded.

To watch one level do one task, with its reply (levels 0-1) or every tool call printed as
it happens (levels 2-5; the transcript keeps the full record), the verify.py verdict and
one cost line:

```bash
python examples/stage_0_bare_model.py          # ... stage_1_system_prompt.py, stage_2_one_tool.py,
python examples/stage_5_sub_agents.py --task swe2   # stage_3_tools.py, stage_4_harness.py
```

## Where results land

`benchmark/ladder/results/<YYYY-MM-DD_HHMM>/` (or `--out`):

- `runs.jsonl`: one JSON line per run, appended and flushed as it finishes: level, task, run,
  seed, passed, verify note, stop reason, steps, tokens, seconds, tool calls and errors,
  nudges, children, files applied, the digest and the wrap-up summary.
- `results.md`: THE TABLE. Rows are the six levels with their "adds" text; columns are one
  `x/k` per task, overall `x/N (p%)`, average steps, tokens and seconds, and a stop-reason
  summary (`answered` is the loop's `done`: the model stopped on its own, right or wrong).
  A second small table gives the delta of each level over the row above it (named in the
  cell), then the fixed conditions and the exact command to reproduce. A run killed
  mid-write leaves a torn last line in `runs.jsonl`; `--resume` and `--report-only` drop
  it with a warning and carry on (a damaged line anywhere else is an error).
- `transcripts/L<level>-<task>-r<run>.jsonl`: every run's full trace (levels 0-1 write one
  `one_shot` record; 2-5 a cornac transcript). `python -m cornac.transcript FILE` prints one.

## The same ladder on a bigger model

The runner is provider-agnostic. To point it at a 27B model served by vLLM (or anything
with an OpenAI-compatible `/v1`), keep every other setting and change only the provider:

```bash
python benchmark/ladder/run_ladder.py --provider openai --base-url http://HOST:8000/v1 \
    --model <served-model-name> --out benchmark/ladder/results/27b
```

Same levels, tasks, seeds, temperature and step budget. The frozen generation settings go
in the request in that server's words: `temperature`, `seed`, `max_tokens` 2048 (the per-step
output cap, Ollama's `num_predict`) and, for a Qwen 3 family model name, thinking off via
`chat_template_kwargs: {"enable_thinking": false}` (vLLM's switch; Ollama's `think: false`).
What the request cannot carry is the context length: `num_ctx 8192` is the server's
`--max-model-len`, so serve the model at 8192 (the loop is told 8192 for context clearing
either way). `presence_penalty` needs no field: the OpenAI default is already 0. Every
option sent is on each run line and in the report's header, so the two tables can be read
side by side: does a stronger model still need the harness, and does it use sub-agents
better?
