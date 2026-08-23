---
name: launch-run
description: Produce the exact copy-paste Colab payload to start a training run on GPU, after verifying the repo state is clean and pushed. Use whenever I say start training, run it on Colab, kick off the run, or similar.
argument-hint: <run_id>
---

# Launch a run on Colab

Delegate to the `colab-runner` subagent. It must:

1. Run `git status --porcelain` and `git log -1 --oneline`. If the tree is dirty,
   **stop** and list the uncommitted files.
2. Confirm `configs/<run_id>.yaml` and `experiments/<run_id>/hypothesis.md` exist.
   If not, tell me to run `/new-experiment` first.
3. Launch via the `colab` CLI: `colab new -s <run_id> --gpu <gpu>`, `colab install -s
   <run_id> -r requirements.txt`, then `colab exec -s <run_id> -f <training entrypoint>`
   (or `colab run --gpu <gpu> -- python -m src.train ...` for a short job that should
   tear itself down). No copy-pasting a cell into a browser tab.
   - Only fall back to printing a `scripts/colab_bootstrap.py` cell for manual paste if
     the `colab` CLI isn't authenticated/available — say so explicitly if you do this.
4. Write `experiments/<run_id>/launch.json` with run ID, commit, config path, GPU,
   timestamp, and whether it went through the CLI or the manual-paste fallback.
5. Tell me the two things to check before walking away: the GPU name from `colab status`,
   and that epoch 1 completed.

Then stop. Do not poll. Use `/check-run` for that.
