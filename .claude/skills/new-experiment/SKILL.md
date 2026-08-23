---
name: new-experiment
description: Scaffold a new, properly bookkept training experiment — config file, run ID, experiment directory, and a stated hypothesis. Use this whenever I want to try a new model, loss, patch size, sampling scheme, or any training change, even if I phrase it casually like "let's try a bigger patch".
argument-hint: <short-name> "<hypothesis>"
---

# Start a new experiment

1. Derive a run ID: `YYYYMMDD-<short-name>` (lowercase, hyphenated, no spaces).
2. Copy the closest existing config in `configs/` — usually the current best run — into
   `configs/<run_id>.yaml`. State in a comment at the top which config it derived from.
3. Change **one thing**. If I asked for several changes at once, say so and ask which
   single variable to isolate, or propose a short ordered sequence of runs instead.
   Do not silently bundle changes; it destroys the ablation.
4. Create `experiments/<run_id>/` containing `hypothesis.md`:
   - **Hypothesis:** what I expect to improve and by roughly how much
   - **Changed from baseline:** exactly one line describing the delta
   - **Falsification:** what result would mean this idea is wrong
   - **Cost:** estimated GPU-hours
5. Append a line to `docs/PROGRESS.md` under today's date.
6. Print the exact launch command and stop. Do not start training from this skill.

If the proposed change has no plausible mechanism for improving lesion-wise Dice or NSD,
say that before scaffolding anything.
