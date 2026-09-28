# The capability ladder — results

Generated 2026-09-27 22:24 by `benchmark/ladder/run_ladder.py` from
54 runs in `runs.jsonl` (35 passed). One JSON line per run; the tables are
rebuilt from the whole file each time (`--report-only`). Each run's full conversation is
in `transcripts/L<level>-<task>-r<run>.jsonl`, which is where to look when a number here
needs explaining.

## Fixed conditions

Model `qwen3:4b-instruct-2507-q4_K_M` via ollama; temperature 0.3; seed 42 + run index
(run i of every level and task uses the same seed); k = 3 runs per level per task;
max_steps 12; generation options as sent on every request: `{"num_ctx": 8192,
"num_predict": 2048, "presence_penalty": 0.0, "seed": 42, "temperature": 0.3}`. Grading
is each task's own verify.py on the workspace copy the level left behind; the task
prompt, the fixture and the grader are the same at every level. Headless: nobody answers
a permission ASK, so ASK is DENY. Every text the model reads at a level — tool
descriptions and system prompt — names only the tools that level has.

The permission policy, identical at every level (a safety property, not a capability):

```json
{
  "default": "deny",
  "tools": {
    "read_file": "allow",
    "write_file": "allow",
    "edit_file": "allow",
    "list_dir": "allow",
    "grep": "allow",
    "run_python": "allow",
    "spawn_agent": "allow",
    "run_bash": {
      "deny": [
        "rm -rf",
        "sudo",
        "mkfs",
        "dd if=",
        ":(){"
      ],
      "default": "allow"
    }
  }
}
```

## The ladder

Same model, same tasks, one more thing per level. The **overall** column is the number
that climbs.

| # | level | adds | swe | swe2 | swe3 | overall | avg steps | avg tokens | avg secs | stop reasons |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | `bare` | nothing: the question and the code, one model call | 3/3 | 3/3 | 0/3 | **6/9 (67%)** | 1.0 | 683 | 4.8 | answered 9 |
| 1 | `prompted` | a system prompt | 3/3 | 2/3 | 3/3 | **8/9 (89%)** | 1.0 | 777 | 5.5 | answered 9 |
| 2 | `one_tool` | one tool, run_bash: the files are no longer pasted in; the agent loop begins here | 0/3 | 0/3 | 0/3 | **0/9 (0%)** | 5.8 | 13,840 | 47.5 | answered 4 · max_steps 3 · error 2 |
| 3 | `tools` | the file tools: read_file, write_file, list_dir, grep, run_python (no syntax check) | 3/3 | 3/3 | 0/3 | **6/9 (67%)** | 7.3 | 13,866 | 28.1 | answered 8 · max_steps 1 |
| 4 | `harness` | edit_file, syntax gate, change report, nudge, repeat note/stop, wrap-up, context clearing | 3/3 | 3/3 | 2/3 | **8/9 (89%)** | 6.8 | 14,879 | 27.7 | answered 8 · stuck 1 |
| 5 | `agents` | sub-agents | 3/3 | 2/3 | 2/3 | **7/9 (78%)** | 7.2 | 19,801 | 43.4 | answered 8 · stuck 1 |

Columns: `#`/level are the level's number and key (benchmark/ladder/levels.py); adds is
the ONE thing that level has and the level below does not; each task column is
passes/runs for that task; overall is passes over all of the level's runs; avg steps is
loop turns per run — one per model call inside the agent loop; the wrap-up call after an
unfinished run and a sub-agent's own calls are counted in tokens, not steps; a level-0/1
run is always one turn; avg tokens is prompt + output tokens per run, summed over every
model call, wrap-up and sub-agents included; avg secs is wall-clock per run, tool time
included; stop reasons is how the level's runs ended — `answered` (the model stopped on
its own and gave a final message, right or wrong: the loop's stop_reason `done` on the
run line), `max_steps` (the loop's budget ran out), `stuck` (the same call gave the same
result three times), `error` (the provider raised).

## What the extra step bought

Each row against the row above it — the next lower level that was run, named in the
cell — so the delta is the passes the added thing bought (or cost) when the two rows
were run the same number of times, and percentage points alone when they were not.
Read it with the transcripts: a delta of one run out of nine is one run.

| # | level | adds | overall | vs the row above |
|---|---|---|---|---|
| 0 | `bare` | nothing: the question and the code, one model call | 6/9 (67%) | baseline |
| 1 | `prompted` | a system prompt | 8/9 (89%) | vs L0: +2 passes (+22 pp) |
| 2 | `one_tool` | one tool, run_bash: the files are no longer pasted in; the agent loop begins here | 0/9 (0%) | vs L1: -8 passes (-89 pp) |
| 3 | `tools` | the file tools: read_file, write_file, list_dir, grep, run_python (no syntax check) | 6/9 (67%) | vs L2: +6 passes (+67 pp) |
| 4 | `harness` | edit_file, syntax gate, change report, nudge, repeat note/stop, wrap-up, context clearing | 8/9 (89%) | vs L3: +2 passes (+22 pp) |
| 5 | `agents` | sub-agents | 7/9 (78%) | vs L4: -1 passes (-11 pp) |

4 run(s) were re-run from scratch after an infrastructure error (Ollama's model runner stopped; see `infra_errors` on the line): L2 swe3 r1, L3 swe3 r2, L5 swe r0, L5 swe3 r2. Only the re-run counts; the crash is recorded, not scored.

## How to reproduce

```bash
cd <repo root>
source .venv/bin/activate          # the runner puts this interpreter's bin dir on PATH itself
python benchmark/ladder/run_ladder.py --levels 0-5 --tasks swe,swe2,swe3 --k 3 --temperature 0.3 --seed-base 42 --provider ollama --model qwen3:4b-instruct-2507-q4_K_M --max-steps 12 --out benchmark/ladder/results/2026-09-27_2158
```

Add `--resume` to continue a stopped matrix or to add a level to this directory;
`--report-only benchmark/ladder/results/2026-09-27_2158` rebuilds this file from `runs.jsonl` without running anything.
