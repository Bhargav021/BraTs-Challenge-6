---
name: explain-this
description: Explain a piece of this project to me at the depth I need to make a decision about it, not just describe the code. Use whenever I ask what does this do, why did we choose that, or explain X, and whenever I need to understand a result well enough to judge it.
argument-hint: <file, function, concept, or run_id>
---

# Explain it to me

Assume I know Python and general deep learning, and that I am still building depth in
medical image segmentation and the BraTS evaluation protocol.

Structure the explanation:

1. **What it is** — one sentence, no jargon.
2. **Why it exists here** — what would break or get worse without it, in terms of the
   challenge metrics (lesion-wise Dice, NSD) or training stability.
3. **How it works** — the mechanism, with a small concrete example (a tiny volume,
   two lesions, actual numbers). Prefer one worked example over three paragraphs.
4. **The tradeoff** — what it costs: compute, memory, generalization, complexity.
5. **What I'd have to decide** — the knobs that are genuinely my call, with your
   recommendation and what evidence would change it.
6. **If you want to go deeper** — at most two pointers (a paper section, a file),
   with the time each takes.

Define each domain term once, in a clause. Do not walk through the code line by line
unless I ask. If something in the code is wrong or suspicious, say so here rather than
explaining it as if it were correct.
