---
name: ablate
description: Design and queue a proper ablation study — one variable at a time, matched seeds and budget, with the comparison table defined before any run starts. Use when I want to know whether a component actually helps, or when several changes are competing for GPU time.
argument-hint: <component to ablate>
---

# Design an ablation

1. Name the baseline run ID everything will be compared against. If there isn't one
   yet, say so — an ablation without a baseline is not an ablation.
2. List the variants, each differing from the baseline in exactly one config key.
   Maximum 4 variants per study. If the request implies more, propose a two-stage plan.
3. Fix what must be held constant across all variants and write it down: seed(s),
   split file, patch size, epochs or step budget, augmentation, evaluation code version.
4. Estimate total GPU-hours. If it exceeds one day of Colab availability, cut variants
   and say which question you're deferring.
5. Write `experiments/ablations/<name>.md` containing:
   - the question in one sentence
   - the variant table (run ID, changed key, value, status)
   - the empty results table, with the six official regions as columns
   - the decision rule: what difference in lesion-wise Dice you'd need to call it real,
     given how noisy this metric is across seeds
6. Scaffold each variant with `/new-experiment` and print the launch order.

State up front how many seeds it would take to distinguish the expected effect from
run-to-run noise. If the honest answer is "more seeds than we can afford", say so —
that is a legitimate finding and it should change what we try next.
