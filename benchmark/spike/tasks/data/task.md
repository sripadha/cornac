# Task: data — total revenue from a CSV

Domain: data analysis. Fixture: `fixture/sales.csv`, forty rows with the columns
product, units, price.

## Prompt

The workspace contains a file named sales.csv with the columns product, units and
price. Compute the total revenue across all rows, where a row's revenue is units
multiplied by price. Report the total as a number with exactly two decimal places.

## Pass rubric

The final answer contains the exact total, 121764.46. A thousands separator or a
leading dollar sign is fine: 121,764.46 and $121764.46 both pass. Rounded or
truncated values (121764, 121764.5, 121764.460) fail. Only the agent's final message
is checked, not its tool output: computing the right number and then reporting a
different one is a failure, because the final message is what a user would see.

## Why this task

The total is not a round number and cannot be guessed from the prompt. Passing takes
one tool call that actually computes (run_python, or run_bash with awk). The fixture
started with six rows, and that was too few: a verbose model read the file and then
multiplied the six rows by hand in a long prose reply — correctly — so it passed
without ever computing with a tool, and `answered_without_tools` stayed False because
the one read_file call counted as tool use. Forty rows of non-round prices put hand
arithmetic out of reach, so the task now does separate "used a tool to compute" from
"guessed" or "worked it out in prose". `tools_used` in runs.jsonl records whether
run_python or run_bash was actually invoked.
