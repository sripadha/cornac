# Model spike results

Generated 2026-09-22 01:39 by `benchmark/spike/run_spike.py` from
45 runs in `runs.jsonl` (one JSON line per run; the table is rebuilt from the
whole file each time the script runs). Each run's full conversation is in
`transcripts/`, which is where to look when a number here needs explaining.

| model | tool probe | swe | swe2 | swe3 | overall | tool-error rate | no-tool answers | announced-then-stopped | errors | ctx overflow | avg steps | avg tokens | avg secs | GPU % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `hf.co/unsloth/Qwen3.5-4B-GGUF:Q8_0` | ok | 3/3 | 3/3 | 3/3 | 9/9 (100%) | 0/57 (0%) | 0 | 0 | 0 | 0 | 6.4 | 13060 | 98.2 | 55 |
| `hf.co/unsloth/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M` | **NO** | 0/3 | 0/3 | 0/3 | 0/9 (0%) | n/a | 9 | 9 | 0 | 0 | 1.0 | 1146 | 15.0 | 79 |
| `qwen3:4b-instruct-2507-q4_K_M` | ok | 1/3 | 3/3 | 0/3 | 4/9 (44%) | 0/87 (0%) | 0 | 1 | 0 | 0 | 10.2 | 26534 | 38.1 | 100 |
| `qwen3.5:4b` | ok | 3/3 | 1/3 | 3/3 | 7/9 (78%) | 0/70 (0%) | 0 | 0 | 0 | 0 | 7.2 | 16816 | 25.2 | 100 |
| `qwen3:8b` | ok | 1/3 | 0/3 | 0/3 | 1/9 (11%) | 0/54 (0%) | 0 | 8 | 0 | 0 | 7.0 | 11229 | 60.8 | 64 |

> **Warning:** `hf.co/unsloth/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M` emitted no tool call in the probe (reply began: "<xml> {"name": "get_current_time", "arguments": {"timezone": "Asia/Tokyo"}} </xml>"). A model that cannot emit tool calls through Ollama is disqualified for this harness; its rows are kept for the record.

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
transcript before trusting any one count); errors is runs that died on a provider
error (an Ollama 500, a timeout; there are no retries, so one timeout is one error)
and count as fails; ctx overflow is runs in which some call's prompt plus output
reached num_ctx, so Ollama silently truncated the prompt and any failure after that
point is a budget problem rather than a tool-calling one; avg tokens is input +
output per run; GPU % is the share of the model that Ollama placed on the GPU.

## Model facts

| model | pull size | placement (ollama ps) | tokens/sec | tool probe | probe reply (raw, first 80 chars) |
|---|---|---|---|---|---|
| `hf.co/unsloth/Qwen3.5-4B-GGUF:Q8_0` | 5.2 GB | 45%/55% CPU/GPU | 7.5 | ok | The current time in Tokyo is 2:04 PM on Tuesday, September 22, 2026 (JST). |
| `hf.co/unsloth/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M` | 4.7 GB | 21%/79% CPU/GPU | 17.8 | **NO** | <xml> {"name": "get_current_time", "arguments": {"timezone": "Asia/Tokyo"}} </x… |
| `qwen3:4b-instruct-2507-q4_K_M` | 2.5 GB | 100% GPU | 44.7 | ok | The current time in Tokyo is 2026-09-22 14:20:51 JST. |
| `qwen3.5:4b` | 3.4 GB | 100% GPU | 31.8 | ok | The current time in Tokyo is **14:26:44** on September 22, 2026 (JST). |
| `qwen3:8b` | 5.2 GB | 36%/64% CPU/GPU | 7.2 | ok | The current time in Tokyo is 14:31:10 JST on September 22, 2026. |

## How to reproduce

```bash
cd <repo root>
source .venv/bin/activate          # pytest must be installed (the swe tasks run it)
ollama pull hf.co/unsloth/Qwen3.5-4B-GGUF:Q8_0
ollama pull hf.co/unsloth/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama pull qwen3.5:4b
ollama pull qwen3:8b
python benchmark/spike/run_spike.py --models hf.co/unsloth/Qwen3.5-4B-GGUF:Q8_0,hf.co/unsloth/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M,qwen3:4b-instruct-2507-q4_K_M,qwen3.5:4b,qwen3:8b --tasks swe,swe2,swe3 --k 3 --temperature 0.3 --seed-base 42 --out benchmark/spike/results_round2
```

Settings that make runs comparable: temperature 0.3, seed 42 + run index,
num_ctx 8192, num_predict 2048 per step, max_steps 12, one HTTP
attempt per call (no retries), one shared policy and tool set, think=False for
every Qwen 3 family tag. Runs append to `runs.jsonl`; delete the results directory
for a clean slate.
