# Ollama & the Python Toolchain — Environment Lessons

Setup-time explanations worth keeping: what Ollama actually is (and why we use it
instead of running a HuggingFace download directly), and how `uv`, `.venv`, and `pip`
relate. These decisions are recorded in the plan; this file preserves the *understanding*.

---

## Part 1 — What is Ollama, really?

### A model file is inert

"Downloading a model" means downloading **a big bag of numbers** — the weights (~15GB in
FP16 for Qwen 2.5 7B). Weights alone can't do anything, the way a music score is just
ink until someone plays it. To get "type a question, get an answer" you need four layers:

| Layer | Job | Analogy |
|---|---|---|
| **Weights** | the learned numbers (the knowledge) | the score |
| **Inference engine** | runs the math: text in → next token, repeat | the musician |
| **Tokenizer + chat template** | text ↔ numbers, formats the conversation the way *this* model expects | the notation conventions |
| **Server / API** | lets your code send a request and get a reply | the concert hall's box office |

A HuggingFace download gives you mainly the weights (+ tokenizer). **Ollama bundles all
four into one command** — it wraps `llama.cpp` (a fast C++ inference engine) with a
model registry, automatic quantization, GPU offload, and an HTTP server.

### Why not just `transformers` from HuggingFace?

The direct path on an RTX 2060 (6GB VRAM):

```python
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
                                             torch_dtype=torch.float16, device_map="cuda")
```

**crashes out of memory** — FP16 weights are ~15GB and the card has 6GB. Making it fit
means picking a quantization method (GPTQ? AWQ? bitsandbytes?), fighting CUDA versions,
and hand-tuning layer offload — an afternoon per model, before the first answer. Plus you
apply the chat template yourself, and getting it subtly wrong produces garbage silently.

The Ollama path:

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M    # ALREADY-quantized 4-bit, ~4.7GB — fits 6GB
```

Decoding the tag: `qwen2.5` family · `7b` params · `instruct` chat-tuned ·
`q4_K_M` = 4-bit quantization, the quality/size sweet spot.

### Why Ollama fits this project specifically

1. **It exposes an HTTP API** (`localhost:11434`) — the same *shape* of interaction as
   Anthropic's cloud API. That symmetry is why `OllamaProvider` and `AnthropicProvider`
   are structural twins ("build request → POST → parse reply") and why the Provider seam
   is elegant rather than forced.
2. **Structured tool calls** — Ollama parses the model's native tool-call syntax and
   returns a `tool_calls` field, so the provider stays thin.
3. **Reproducibility for readers** — the benchmark's setup is one `ollama pull`, not
   "install these 8 packages and hope CUDA matches."
4. **The daemon keeps the model warm.** First call pays ~16s to load 4.7GB into VRAM;
   subsequent calls run in ~1s. Crucial when a benchmark makes many calls back-to-back.
5. **Verified on this machine:** GPU detected, VRAM 761MiB → 4803MiB when loaded (~4GB
   jump = the model living on the GPU, not the CPU).

### The honest tradeoff

Ollama **hides the inference internals** — KV cache, attention, quantization math,
batching, sampling. For *this* project that's correct: the harness is the layer *above*
inference, and the boundary between "our code" and "their code" is exactly the HTTP/SDK
line. The inference deep-dive (paged attention, speculative decoding, manual
quantization…) is deliberately a separate future project — bolting it into the harness
would dilute both. Middle ground if ever wanted: `llama-cpp-python` in-process (you
configure quant + GPU offload yourself, ~50 lines) — a v2 footnote, not a v1 goal.

> **One-liner:** the model is sheet music; Ollama is the musician, instrument, and
> concert hall. We're building the layer that *directs the performance* — not the
> instrument.

---

## Part 2 — `uv` vs `.venv` vs `pip`

### The problem venvs solve

Two projects need different versions of the same library (httpx 0.20 vs 0.27). One
shared pile of packages can't hold both → give each project its **own private box** of
packages. That box is a **virtual environment**.

### The three things, and which is which

| Thing | What it is | Analogy |
|---|---|---|
| **`.venv/`** | the box itself — a *folder* holding a private `bin/python` and a `site-packages/` with this project's libraries | the bathroom |
| **`uv`** | the tool that *builds and fills* boxes (and can install Python versions) | the plumber |
| **`pip`** | the traditional tool that only *fills* boxes | the plumber's older, slower fitting-installer |

You don't need the plumber present to use the bathroom: day-to-day you just **activate**
the venv. You call `uv` only to *change* the environment.

### The two jobs, and who does them

| Tool | Job 1: make the venv | Job 2: install packages | Bonus: install Python itself |
|---|---|---|---|
| `python -m venv` | ✅ | ❌ | ❌ |
| `pip` | ❌ | ✅ | ❌ |
| **`uv`** | ✅ | ✅ (10–100× faster) | ✅ |

The bonus column decided it for us: the system had Python 3.10 (EOL Oct 2026) and we
wanted 3.12. `uv python install 3.12` fetched it into the home directory, **no sudo** —
`python -m venv` can't do that at all. One fast Rust tool replaces `pyenv` + `venv` + `pip`.

### The timeline that untangles the confusion

```
ONCE, at setup (uv ACTS):
    uv python install 3.12         # fetch Python 3.12 (no sudo)
    uv venv --python 3.12          # build .venv/
    uv pip install -e ".[dev]"     # fill it with the project + deps

EVERY DAY (use the box; uv not involved):
    source .venv/bin/activate      # prompt shows (agentai)
    python examples/...            # runs the box's python

WHEN ADDING A DEPENDENCY (uv ACTS again):
    uv pip install pyyaml          # drop one more package into the existing box
```

`uv run python script.py` is the one blur: uv briefly "steps into" the venv for a single
command — an alternative to activating, same effect.

### Where Ollama fits in this picture: nowhere

Ollama is a **system-level program** (like `git` or a database server), installed once
with its own installer, running as a daemon. It doesn't care whether a venv is active.
The venv matters only for the *Python code that talks to* Ollama over HTTP.

Declared in `pyproject.toml`: `requires-python = ">=3.10"` — we *develop* on 3.12 but
the published package installs for the widest audience. Develop modern, ship broad.
