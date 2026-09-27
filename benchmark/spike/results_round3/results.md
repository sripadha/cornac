# Model spike results

Generated 2026-09-22 22:43 by `benchmark/spike/run_spike.py` from
27 runs in `runs.jsonl` (one JSON line per run; the table is rebuilt from the
whole file each time the script runs). Each run's full conversation is in
`transcripts/`, which is where to look when a number here needs explaining.

| model | tool probe | swe | swe2 | swe3 | overall | tool-error rate | no-tool answers | announced-then-stopped | nudges | errors | ctx overflow | avg steps | avg tokens | avg secs | GPU % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `qwen3:4b-instruct-2507-q4_K_M` | ok | 3/3 | 3/3 | 3/3 | 9/9 (100%) | 0/52 (0%) | 0 | 0 | 1 | 0 | 0 | 6.9 | 15455 | 20.9 | 100 |
| `qwen3:8b` | ok | 3/3 | 2/3 | 3/3 | 8/9 (89%) | 15/76 (20%) | 0 | 0 | 4 | 0 | 0 | 9.7 | 20791 | 71.4 | 64 |
| `qwen3.5:4b` | ok | 3/3 | 3/3 | 3/3 | 9/9 (100%) | 0/59 (0%) | 0 | 0 | 0 | 0 | 0 | 6.2 | 15974 | 23.5 | 100 |

Columns: tool probe is the pre-run check that the model can emit a tool call at all
(one tiny task with only get_current_time registered; `tool_capable` on every run
line, the raw reply in `models.json` and the facts table below; a **NO** disqualifies
the model for this harness and its rows are kept only for the record); each task
column is passes/runs for that task; data via tool (shown when the data task ran) is
data runs in which run_python or run_bash was invoked at all (a data pass without
one was arithmetic in prose; `tools_used` in runs.jsonl lists every run's tools);
overall is passes over all runs; tool-error rate is bad tool calls over all tool
calls, counting both kinds — hard errors, where the policy denied the call or the
tool raised (`n_tool_errors` in runs.jsonl), and soft errors, where the tool ran and
reported failure in its text, "Error: not a file" and the like (`n_soft_errors`);
no-tool answers is runs where the model answered without calling any tool (a
guess); announced-then-stopped is a HEURISTIC: runs that ended with a final message
carrying no tool call, did not pass, and whose final text contains a phrase like
"let me", "let's", "I will", "I'll", "now I" or "next" — the model announcing a next
step and then ending its turn instead of taking it (a phrase match, so read the
transcript before trusting any one count); nudges is the total number of times the
agent nudged the model on after such an announcement (`nudges` on each run line,
0 on lines that predate the field; a run that passed after a nudge is a pass the
harness bought); errors is runs that died on a provider
error (an Ollama 500, a timeout; there are no retries, so one timeout is one error)
and count as fails; ctx overflow is runs in which some call's prompt plus output
reached num_ctx, so Ollama silently truncated the prompt and any failure after that
point is a budget problem rather than a tool-calling one; avg tokens is input +
output per run; GPU % is the share of the model that Ollama placed on the GPU. A
model cell with "(options: ...)" is a row run with that `--options` override (see
below); the same tag without it is a separate row.

## Model facts

| model | pull size | Modelfile PARAMETERs (ollama show --modelfile) | placement (ollama ps) | tokens/sec | tool probe | probe reply (raw, first 80 chars) |
|---|---|---|---|---|---|---|
| `qwen3:4b-instruct-2507-q4_K_M` | 2.5 GB | top_k 20; top_p 0.8; repeat_penalty 1; stop <\|im_start\|>; stop <\|im_end\|>; temperature 0.7 | 100% GPU | 41.9 | ok | The current time in Tokyo is 2026-09-23 11:25:37 JST. |
| `qwen3:8b` | 5.2 GB | repeat_penalty 1; stop <\|im_start\|>; stop <\|im_end\|>; temperature 0.6; top_k 20; top_p 0.95 | 36%/64% CPU/GPU | 6.5 | ok | The current time in Tokyo is 2026-09-23 11:29:09 JST. |
| `qwen3.5:4b` | 3.4 GB | top_p 0.95; presence_penalty 1.5; temperature 1; top_k 20 | 100% GPU | 33.0 | ok | The current time in Tokyo is 11:39 AM on September 23, 2026 (JST). |

Modelfile PARAMETERs are the sampling settings the tag itself bakes in (`ollama show
<tag> --modelfile`; `modelfile_parameters` in `models.json`) and applies to every
request that does not set the same key; a request option — the runner's temperature,
or anything in `--options` — overrides the PARAMETER of the same name.

## How to reproduce

```bash
cd <repo root>
source .venv/bin/activate          # pytest must be installed (the swe tasks run it)
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama pull qwen3:8b
ollama pull qwen3.5:4b
python benchmark/spike/run_spike.py --models qwen3:4b-instruct-2507-q4_K_M,qwen3:8b,qwen3.5:4b --tasks swe,swe2,swe3 --k 3 --temperature 0.3 --seed-base 42 --out benchmark/spike/results_round3
```

Settings that make runs comparable: temperature 0.3, seed 42 + run index,
num_ctx 8192, num_predict 2048 per step, max_steps 12, one HTTP attempt per call
(no retries), one shared policy and tool set, think=False for every Qwen 3
family tag. No --options override was used. Runs append to `runs.jsonl`; delete
the results directory for a clean slate.
