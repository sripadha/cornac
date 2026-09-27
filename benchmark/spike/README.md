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
- **`announced_then_stopped`** on every line, and an `announced-then-stopped` column in the table. This is a HEURISTIC: true when the run's final assistant message had no tool calls, the task did not pass, and the final text ends on a line that announces a step ("let me", "let's", "I will", "I'll", "I'm going to", "now I", "next step"; case-insensitive). It is a phrase match, no more - it sorts transcripts, it does not judge them. Since round three the phrase test is the agent's own nudge trigger (`announces_next_step` in `cornac/core/agent.py`), so this column and `nudges` count the same thing by construction. Rounds one and two used the runner's own list, which also matched a bare "next" and so flagged some honest summaries; applied to round one's transcripts that list flagged exactly the four 7-8B `swe` failures (and every `qwen3:4b` run, whose leaked chain of thought is all announcements).

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

## Round three: the harness under test, `results_round3/`

The model is frozen (`qwen3.5:4b`; `OllamaProvider` now sends `presence_penalty 0`, `num_predict 2048` and `think=False` for the Qwen 3 family by default, so no `--options` override is needed). What round three measures is the **harness**. Reading the 45 transcripts of round two and the round two-b runs, most coding failures were ours, not the model's:

1. 13 failures were whole-file `write_file` rewrites that dropped untouched functions or mangled a closing `"""` - a one-line fix became a broken file, because rewriting the whole file was the only way to change a file.
2. In 8 runs the broken file was discovered only by the next pytest run, and then the identical broken write was re-sent 3-6 times.
3. 13 runs ended with the model announcing the next step ("Let me fix that and re-run the tests") and then stopping with no tool call.
4. `write_file` reported "Wrote 179 chars" when it had just deleted two functions.
5. Models guessed a cwd or a path (`cd /workspace`, pytest on the wrong file) because the system prompt never said where the workspace was.
6. One model "fixed" a function inside a `run_python` snippet and reported the file changed.

Week 4b closes those gaps in `cornac`, and round three re-runs the same three tasks, same k, same frozen model, to see how many of the failures the scaffolding removes:

    python benchmark/spike/run_spike.py --models qwen3.5:4b --tasks swe,swe2,swe3 --k 3 --temperature 0.3 \
        --seed-base 42 --out benchmark/spike/results_round3

Harness items under test (each a Week 4b change, in `cornac/`, with its own tests):

- **`edit_file`** - a few-lines edit (an exact old-text to new-text replacement) next to the whole-file `write_file`, so a one-line fix no longer means retyping a file the model has just read.
- **A syntax check after writing a Python file** - a write that leaves the file unparseable says so in the tool result, instead of being found by the next pytest and re-sent unchanged.
- **The nudge** - when the model ends its turn by announcing an action it did not take, the agent sends it on once ("you said you would do that; go on") before accepting the text as final. `max_nudges` is an `Agent` setting; the runner leaves it at the default.
- **An honest `write_file` summary** - what changed, not just a character count.
- **`Workspace.describe()`** - one paragraph in the system prompt: the absolute root, that every tool path is relative to it, and that `run_bash`/`run_python` already start there (no `cd`).
- **`run_bash`/`run_python` descriptions** - they say the command runs in the workspace and does NOT edit source files ("to change a file use edit_file or write_file").

What changed in the runner:

- The spike policy allows `edit_file` alongside `write_file`; the tool comes from `default_tools`, like every other built-in.
- The per-run system prompt is the fixed `SYSTEM_PROMPT` plus `Workspace.describe()` for that run's temp directory (`system_prompt_for`). Same paragraph for every model, so it is part of the fixed conditions.
- Every `runs.jsonl` line records `nudges` (how many times the agent nudged the model on; `0` on older lines, which predate the field), the per-run stdout line shows `nudges=N` when it is non-zero, and the table has a `nudges` column (summed over the row's runs). A pass that needed a nudge is a pass the harness bought, which is the thesis in one number.
- The `Agent` is built with only what the spike must pin (provider, tools, system prompt, `max_steps`, policy); the nudge limit, hooks and approver stay at the `Agent` defaults, so a run measures the harness as shipped.
- The `options` field on each line is the dict the provider actually sent: its frozen defaults, then the runner's `temperature`/`num_ctx`/`seed`/`num_predict`, then `--options` last (`SpikeProvider.request_options` builds both the request and the record). Since the provider now sends `think=False` for the Qwen 3 family itself, the runner's only job there is the fallback: drop the flag for a tag whose template rejects it and re-send once.
- The default `--models` is the frozen tag. Rounds one and two named their candidates by hand, and their commands above still do.

The tasks are unchanged, so a round-three row reads directly against round two-b's `qwen3.5:4b` row (9/9 at ~20 s a run): the interesting numbers are steps, tool-error rate, `announced-then-stopped` and `nudges`, and whether `edit_file` shows up in `tools_used` at all.
