# Task: research — find the note that discusses gradient checkpointing

Domain: research (retrieval over local notes). Fixture: `notes/` with five short
research notes from training a small language model.

## Prompt

The directory notes/ contains several short research notes. Find which note discusses
gradient checkpointing and report two things: the name of that file, and the memory
saving it claims. Answer plainly, in the form "File: <file name>. Memory saving:
<value>".

## Pass rubric

The final answer contains the file name `memory-experiments.txt` AND the number 37
(line 4 of that note says gradient checkpointing "cut memory by 37%"). Both are
required: the file name alone means the agent found the note but did not read it;
37 alone could be a lucky guess. The file name may be given with or without the
`notes/` prefix. The number must stand on its own: 37, 37%, "37 percent" and a
sentence-final "37." all count, while 37.5, 2.37, 0.37 and 137 do not. If the answer
uses the requested `File:` form, that slot must name only the right note: "File:
data-loading.txt or memory-experiments.txt" is a hedge, not an answer, and fails.
Mentioning another note elsewhere, to explain why it was ruled out, is fine.

## Why this task

A grep for "checkpoint" hits two notes (data-loading.txt talks about saving model
checkpoints), so the agent has to read and tell the two meanings apart. The correct
sentence also carries a decoy number (a 19% slowdown), so the agent has to report the
memory figure, not just any percentage it saw. No other note contains the number 37.
