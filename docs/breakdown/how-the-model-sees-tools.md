# How the Model Sees Tools — Two Q&A Lessons

Two lessons that came out of hands-on experiments while testing the Week 1 skeleton on
live Qwen 2.5 7B. Both matter directly for the Week 5 benchmark design.

---

## Lesson 1 — Does the model need tools described in the system prompt?

**The question:** our demo's system prompt said *"…use the get_current_time tool rather
than guessing."* Is that how the model knows the tool exists — or would it find the tool
by itself?

**Answer: tools reach the model through TWO separate channels, and only one of them is
load-bearing.**

### Channel 1 — the formal `tools` parameter (the real mechanism)

Every call to `provider.complete(messages, schemas)` sends the structured tool menu from
`registry.schemas()` in the API's dedicated `tools` field:

```json
"tools": [{"type": "function", "function": {
    "name": "get_current_time",
    "description": "Return the current date and time in the given IANA timezone...",
    "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}}}}]
```

**This is how the model knows a tool exists, what it's called, and what arguments it
takes.** It is always sent, independent of the system prompt. The model literally cannot
emit a valid `ToolCall` for a tool this channel didn't describe.

### Channel 2 — the system prompt (a behavioral hint only)

*"…use the get_current_time tool rather than guessing"* is plain-English **advice**
about *when* to prefer the tool. It teaches nothing about the tool's existence or API.

### The proof (run live)

Same agent, system prompt reduced to just `"You are a helpful assistant."` — zero
mention of any tool:

```
System prompt: 'You are a helpful assistant.'
Question:     'What is the current time in India?'

FINAL ANSWER: The current time in India (IST) is 22:52:17.
Tool calls the model made:
  -> get_current_time({'timezone': 'Asia/Kolkata'})
```

The model **discovered and called the tool entirely on its own** — from Channel 1. It
even chose the correct IANA timezone string. (Repeated with the hint commented out and
questions about France/India: the tool call happened every time.)

### So what is the system-prompt hint for?

1. **Steering *when* to use a tool** — weak models sometimes "guess" from training data
   instead of reaching for an available tool. The nudge improves reliability; it doesn't
   grant awareness.
2. **Choosing *among* tools** — with 9 tools, prompt guidance like "prefer grep over
   reading whole files" or "use run_python for any calculation; do not guess numbers"
   shapes tool *choice*.

> **The menu/waiter rule:** the schema is the **menu** — the model can't order a dish it
> was never shown. The system prompt is the **waiter's recommendation** — helpful, but
> you can dine without it. You can never dine without a menu.

### One nuance for the project thesis

*How* Channel 1 works depends on training. Tool-trained models (Qwen 2.5, Claude,
GPT-4) were fine-tuned to read the `tools` parameter natively and emit structured calls
— that's why Qwen "just knew." Models that weren't tool-trained need tool descriptions
stuffed into the prompt text and their output parsed by hand (how early frameworks did
it). Part of what makes a model "agent-capable" is precisely this training — one reason
we chose Qwen 2.5 7B over a smaller model for the benchmark.

---

## Lesson 2 — Run-to-run variance: wording wobbles, behavior doesn't

**The observation:** two runs of the France-time question produced differently-worded
answers ("…is 19:22 CEST (Central European Summer Time)" vs "…is 19:25:32 CEST"), and it
*looked* like commenting out the system-prompt hint caused the difference.

**It didn't. Two confounds were in play:**

1. **The input actually changed between runs.** The runs were minutes apart, so
   `get_current_time` returned different text — and the tool result is part of the
   conversation the model reads. Different input → different phrasing. 
2. **Local models aren't perfectly reproducible even at `temperature: 0`.** Parallel GPU
   floating-point arithmetic can sum in different orders, nudging borderline token
   choices. Identical input can still produce slightly different wording.

### The proof (identical setup, run twice back-to-back, nothing changed)

```
RUN 1: tool_call={'timezone': 'Europe/Paris'}
       final  ='The current time in France (CEST) is 19:29.'
RUN 2: tool_call={'timezone': 'Europe/Paris'}
       final  ='The current time in France (Europe/Paris) is 19:29 CEST (Central European Summer Time).'
```

Same prompt, same everything — the *phrasing* still wobbled. But look at what stayed
rock-solid in every run: **the tool call** (`get_current_time` with
`{'timezone': 'Europe/Paris'}`).

### The two takeaways

**For the benchmark (Week 5):**
- ❌ Don't grade by exact string match on the final answer — it wobbles and produces
  false failures.
- ✅ Grade by *behavior and facts*: did it call the right tool, does the answer contain
  the right values. That's the stable signal. (This is also why the benchmark runs each
  task k=3 times.)

**For experiments in general:** change **one variable at a time**. The original
comparison changed the prompt *and* the wall-clock time *and* rode on inherent variance
— which made an unrelated wording difference look like an effect. The untangling method:
run the identical thing twice first, to measure the noise floor, before attributing any
difference to your change.
