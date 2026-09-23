# Model spike results

Generated 2026-09-22 00:12 by `benchmark/spike/run_spike.py` from
27 runs in `runs.jsonl` (one JSON line per run; the table is rebuilt from the
whole file each time the script runs). Each run's full conversation is in
`transcripts/`, which is where to look when a number here needs explaining.

| model | data | swe | research | data via tool | overall | tool-error rate | no-tool answers | errors | ctx overflow | avg steps | avg tokens | avg secs | GPU % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `qwen2.5:7b-instruct-q4_K_M` | 3/3 | 1/3 | 0/3 | 3/3 | 4/9 (44%) | 0/29 (0%) | 0 | 0 | 0 | 4.2 | 6347 | 21.3 | 79 |
| `qwen3:8b` | 3/3 | 1/3 | 3/3 | 3/3 | 7/9 (78%) | 0/30 (0%) | 0 | 0 | 0 | 4.3 | 5638 | 29.6 | 64 |
| `qwen3:4b` | 0/3 | 0/3 | 0/3 | 0/3 | 0/9 (0%) | 0/2 (0%) | 8 | 0 | 0 | 1.1 | 3526 | 45.5 | 100 |

Columns: each task column is passes/runs for that task; data via tool is data runs
in which run_python or run_bash was invoked at all (a data pass without one was
arithmetic in prose; `tools_used` in runs.jsonl lists every run's tools); overall is
passes over all runs; tool-error rate is bad tool calls over all tool calls, counting
both kinds — hard errors, where the policy denied the call or the tool raised
(`n_tool_errors` in runs.jsonl), and soft errors, where the tool ran and reported
failure in its text, "Error: not a file" and the like (`n_soft_errors`); no-tool
answers is runs where the model answered without calling any tool (a guess); errors
is runs that died on a provider error (an Ollama 500, a timeout; there are no
retries, so one timeout is one error) and count as fails; ctx overflow is runs in
which some call's prompt plus output reached num_ctx, so Ollama silently truncated
the prompt and any failure after that point is a budget problem rather than a
tool-calling one; avg tokens is input + output per run; GPU % is the share of the
model that Ollama placed on the GPU.

## Model facts

| model | pull size | placement (ollama ps) | tokens/sec |
|---|---|---|---|
| `qwen2.5:7b-instruct-q4_K_M` | 4.7 GB | 21%/79% CPU/GPU | 14.9 |
| `qwen3:8b` | 5.2 GB | 36%/64% CPU/GPU | 7.4 |
| `qwen3:4b` | 2.5 GB | 100% GPU | 49.7 |

## How to reproduce

```bash
cd <repo root>
source .venv/bin/activate          # pytest must be installed (the swe task runs it)
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull qwen3:8b
ollama pull qwen3:4b
python benchmark/spike/run_spike.py --models qwen2.5:7b-instruct-q4_K_M,qwen3:8b,qwen3:4b --k 3 --out benchmark/spike/results
```

Settings that make runs comparable: temperature 0, seed 42 + run index, num_ctx
8192, num_predict 2048 per step, max_steps 12, one HTTP
attempt per call (no retries), one shared policy and tool set, think=False for
Qwen 3. Runs append to `runs.jsonl`; delete the results directory for a clean slate.
