---
name: launch-run
description: Produce the exact copy-paste Colab payload to start a training run on GPU, after verifying the repo state is clean and pushed. Use whenever I say start training, run it on Colab, kick off the run, or similar.
argument-hint: <run_id>
---

# Launch a run on Colab

Delegate to the `colab-runner` subagent. It must:

1. Run `git status --porcelain` and `git log -1 --oneline`. If the tree is dirty,
   **stop** and list the uncommitted files — Colab pulls from git and will not see them.
2. Confirm `configs/<run_id>.yaml` and `experiments/<run_id>/hypothesis.md` exist.
   If not, tell me to run `/new-experiment` first.
3. Render `scripts/colab_bootstrap.py` with the run ID, config path, branch, and commit,
   and print it as ONE fenced block I can paste into a single Colab cell.
4. Write `experiments/<run_id>/launch.json` with run ID, commit, config path, timestamp,
   and the Drive mirror path.
5. Tell me the two things to check in Colab before walking away: the GPU name printed in
   the header, and that epoch 1 completed.

Then stop. Do not poll. Use `/check-run` for that.
