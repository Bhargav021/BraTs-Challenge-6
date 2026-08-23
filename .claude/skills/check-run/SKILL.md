---
name: check-run
description: Read the live CLI output and metrics of a Colab training run from the Drive mirror and report status, trend, and any errors. Use whenever I ask how the run is doing, is it done yet, did it crash, or what the loss is doing.
argument-hint: <run_id>
---

# Check a running job

Delegate to the `colab-runner` subagent so the raw log stays out of this conversation.

On this machine (`configs/paths.yaml` → `read_channel: mcp`), it reads `train.log` and
`metrics.jsonl` for the run from `/content/drive/MyDrive/brats_runs/<run_id>/` via the
`mcp__claude_ai_Google_Drive__*` tools (account: `drive_account` in paths.yaml), not
`scripts/tail_run.py` — there is no local mount to run that script against. It reports
back only:

- **Alive?** last log write time vs now; GPU name from the log header
- **Progress:** epoch N of M, seconds/epoch, projected finish time
- **Metrics:** best val metric so far and at which epoch; per-class val Dice for the
  last epoch; the trend across the last 5 epochs (improving / flat / diverging)
- **Anything matching** `Traceback|Error|CUDA out of memory|nan|WARN` — quoted verbatim
- **Verdict:** healthy / stalled / crashed / finished, and the single recommended action

Rules:
- Quote the log. Never paraphrase a number out of it.
- Drive sync lags by a minute or two. Do not declare a run dead from one stale read —
  say "no new output in X minutes" and offer to re-check.
- If the run crashed, give me the resume command, not a rewritten training script.
