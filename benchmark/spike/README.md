# Model spike — choose the benchmark model on evidence, then freeze it

The benchmark runs the whole capability ladder on ONE local model, and that choice has to be made once, not revisited every week. This spike is the decision: the same small offline tasks, run k times on each candidate model, graded by a `verify.py` per task. The model that calls tools most reliably wins and is then frozen in the plan.

Run it (Ollama up, venv active, each model already pulled):

    python benchmark/spike/run_spike.py                                  # every task, k=3
    python benchmark/spike/run_spike.py --models qwen3:4b --k 1          # one model, one run per task

Output goes to `--out` (default `benchmark/spike/results/`): `runs.jsonl` (one line per run, the raw record), `models.json` (pull size, GPU placement, the tool-call probe) and `results.md` (the table, plus a how-to-reproduce block). Runs append, so a model pulled later can be added without rerunning the others.

When a row needs explaining, read `<out>/transcripts/<model>__<task>__run<i>.json`: the full conversation of that run, tool calls and results included. Keep it - a replay with the same seed does not reliably retrace the same trajectory. The `<model>` part of the file name is the tag with every character outside `[A-Za-z0-9._-]` replaced by `_` (`hf.co/unsloth/Qwen3-4B-GGUF:Q8_0` becomes `hf.co_unsloth_Qwen3-4B-GGUF_Q8_0`); the raw tag is what goes to Ollama and what the table shows.

Tasks live in `tasks/<name>/`: `task.md` holds the exact prompt and the pass rubric, `fixture/` the pristine workspace (copied fresh for every run), `verify.py` the grader.

## Round one: `results/` (frozen)

Three tasks, one per benchmark domain (`data`, `swe`, `research`), on `qwen2.5:7b-instruct-q4_K_M`, `qwen3:8b` and `qwen3:4b`, k=3 at temperature 0. Its results directory is not to be appended to. Four lessons drove round two:

1. At temperature 0 the three seeds produced byte-identical runs, so k=3 measured nothing.
2. The tag `qwen3:4b` is the THINKING variant, whose template ignores `think=False`; the fair 4B tag is `qwen3:4b-instruct-2507-q4_K_M`.
3. All four 7-8B `swe` failures were the model announcing "let me fix it and re-run" and then ending its turn with no tool call.
4. `grep(path=<file>)` returned "no matches" for a file path (a harness bug, since fixed in `cornac`), so some coding failures were partly ours.

## Round two: coding focus, `results_round2/`

The project has refocused on coding, so round two runs coding tasks only, with sampling switched on:

    python benchmark/spike/run_spike.py --tasks swe,swe2,swe3 --k 3 --temperature 0.3 \
        --models qwen2.5:7b-instruct-q4_K_M,qwen3:8b,qwen3:4b-instruct-2507-q4_K_M \
        --out benchmark/spike/results_round2

Two new coding tasks, same rigor as `swe` (test file must be AST-identical to the fixture's; pytest runs with `--noconftest -o addopts=` on the named file and must report the exact clean `N passed`; plus a `python -c` behavior check of the fixed function on inputs the tests never mention, so a hardcoded answer fails). Each verifier was checked offline before any model saw it: pristine copy FAIL, hand-fixed copy PASS, edited test FAIL, `pytest.ini`/`conftest.py` cheat FAIL, hardcoded return FAIL.

- `swe2` - the bug is a boundary condition in a helper (`utils.within`, an open interval where a closed one is documented) in a second module, called by the function the failing test exercises (`grades.normalize`). The traceback ends in `grades.py`; the model has to follow the call into `utils.py`. 4 tests, 1 failing.
- `swe3` - a different bug class: a mutable default argument (`def add_item(item, items=[])`) that leaks state between calls. 3 tests; the failing one calls the function twice. The behavior check also rejects a "fix" that empties the caller's list.

Neither prompt names the buggy function or file: "one of the tests fails; find the bug, fix it in the source, do not edit the test file."

New runner options (existing flags keep their meaning):

- `--temperature` (float, default 0.0) is passed to Ollama; `--seed-base` (int, default 42) gives run i the seed `seed_base + i`. Both are recorded on every `runs.jsonl` line (`temperature`, `seed`), and the how-to-reproduce block says if a file mixes temperatures.
- Model tags may contain `/` and `:` (`hf.co/unsloth/...:Q8_0`); only file names are sanitized. `think=False` is sent for any tag whose lowercase name contains `qwen3` (that covers `qwen3`, `qwen3.5` and the unsloth GGUFs); if the daemon rejects the flag for a tag, it is dropped for that tag and the request re-sent once.
- **Tool-call probe.** Before a model's runs, one tiny agent task is sent with ONLY `get_current_time` registered ("What is the current time in Tokyo? Use the tool."). `tool_capable` is whether ANY tool call was emitted. A model that cannot emit tool calls through Ollama is disqualified for this harness; its tasks still run so the rows exist, but every line carries `tool_capable: false`, the table's `tool probe` column says **NO** with a warning under the table, and `models.json` (and the facts table) keep the raw text of the probe reply, first 200 characters, which is where leaked `<think>` text shows. The probe's own transcript is `transcripts/<model>__probe.json`.
- **`announced_then_stopped`** on every line, and an `announced-then-stopped` column in the table. This is a HEURISTIC: true when the run's final assistant message had no tool calls, the task did not pass, and the final text contains a phrase like "let me", "let's", "I will", "I'll", "now I" or "next" (case-insensitive). It is a phrase match, no more - it sorts transcripts, it does not judge them. Applied to round one's transcripts it flags exactly the four 7-8B `swe` failures (and every `qwen3:4b` run, whose leaked chain of thought is all announcements).

Columns in `results.md`: `tool probe` is the probe verdict (`ok`, **NO**, or `not probed` for round-one lines); each task column is passes/runs; `data via tool` (shown only when the data task ran) is data runs in which run_python or run_bash was invoked at all; `overall` is passes over all runs; `tool-error rate` is bad tool calls over all tool calls, counting hard errors (the policy denied the call or the tool raised; `n_tool_errors`) and soft errors (the tool ran and reported failure in its text, such as "Error: not a file"; `n_soft_errors`); `no-tool answers` counts runs that answered without touching a tool (a guess, even when right); `announced-then-stopped` is the heuristic above; `errors` counts runs that died on a provider error (they count as fails; there are no retries, so one timeout is one error); `ctx overflow` counts runs in which a call's prompt plus output reached num_ctx, after which Ollama silently truncates the prompt and a failure is a budget problem, not a tool-calling one; `avg steps`, `avg tokens` (input + output) and `avg secs` are per-run means; `GPU %` is how much of the model Ollama fit on the GPU. Every model runs with the same per-step output cap (num_predict 2048) so a rambling step ends with a recorded `length` stop instead of eating minutes. The model-facts table adds pull size, tokens/sec (output tokens over wall-clock time, tool time included) and the probe reply.

## Round two-b: the sampling check, `results_round2b/`

Round two left two questions open before freezing on `qwen3.5:4b` (7/9, 25 s a run, all on the GPU). First, its two failures were `write_file` payloads that stopped after the first function of a three-function file the model had just read - and `ollama show qwen3.5:4b --modelfile` shows the tag bakes in `presence_penalty 1.5`, `top_k 20`, `top_p 0.95` and `temperature 1`. A presence penalty that high pushes the model away from tokens already in the context, which is exactly what a file rewrite is made of, so the truncation may be a sampling artifact rather than a model limit. Second, the Unsloth Q8_0 build went 9/9 but at 98 s a run with half the model on the CPU; a Q6_K build (~3.5 GB) may fit the GPU and keep the reliability. Round two-b runs both checks into `results_round2b/`:

    python benchmark/spike/run_spike.py --models qwen3.5:4b --tasks swe,swe2,swe3 --k 3 --temperature 0.3 \
        --seed-base 42 --options '{"presence_penalty": 0}' --out benchmark/spike/results_round2b
    python benchmark/spike/run_spike.py --models hf.co/unsloth/Qwen3.5-4B-GGUF:Q6_K --tasks swe,swe2,swe3 --k 3 \
        --temperature 0.3 --seed-base 42 --out benchmark/spike/results_round2b

New in the runner (existing flags keep their meaning):

- `--options '<JSON object>'` is merged into every request's Ollama `options` AFTER the runner's own `temperature`, `num_ctx`, `seed` and `num_predict`, so it can override them; it overrides the tag's Modelfile as well, since a per-request option beats a `PARAMETER` line. It applies to every model of the invocation. Every `runs.jsonl` line records the merged dict (`options`) and the flag's value by itself (`options_override`, `{}` when the flag was absent). The tables keep one row per (model, override), labelled `` `tag` (options: presence_penalty=0) ``, so a tag run with and without an override is never averaged into one row, and the how-to-reproduce block prints one command per override.
- `models.json` records each tag's Modelfile `PARAMETER` lines (`modelfile_parameters`, read from `ollama show <tag> --modelfile`; `[]` when there are none, as for the HF GGUF imports), and the facts table shows them next to the row, so a number is always read beside the sampling settings that were in force.
