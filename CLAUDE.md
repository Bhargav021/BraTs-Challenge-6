# BraTS-PEDs 2025 — Project Instructions

## What this project is
Pediatric brain tumor segmentation for the BraTS-PEDs 2025 challenge (MICCAI/Synapse).
4 input MRI modalities: T1N, T1C, T2W, T2F. Data is **defaced but NOT skull-stripped** —
validation/test data is also not skull-stripped, so **never train on skull-stripped inputs.**

Labels (voxel-wise, 4 classes + background):
- 1 = ET  (enhancing tumor)
- 2 = NET (non-enhancing tumor)
- 3 = CC  (cystic component)
- 4 = ED  (peritumoral edema)
Derived regions: TC = 1+2+3, WT = 1+2+3+4.

Official metrics: **lesion-wise Dice** and **Normalized Surface Distance (NSD)** over
six regions (ET, NET, CC, ED, TC, WT). Plain voxel-wise Dice is a proxy only — never
report it as if it were the challenge score.

## Repository layout
Target layout (not yet built — see "Current state" below):
- `src/data/` — preprocessing, dataset, sampling, augmentation
- `src/models/` — architectures
- `src/losses/` — loss functions
- `src/metrics/` — lesion-wise Dice, NSD, evaluation harness
- `src/train.py` — single entry point for training, config-driven
- `src/infer.py` — sliding-window inference + post-processing
- `configs/*.yaml` — one file per experiment; never hardcode hyperparameters in code
- `experiments/<run_id>/` — config snapshot, logs, metrics.jsonl, checkpoints pointer
- `docs/` — PROGRESS.md, DECISIONS.md, OPEN-QUESTIONS.md, literature/
- `scripts/` — Colab bridge and log tailing utilities

### Current state (2026-08-22)
The repo is still three flat, overlapping scripts/notebooks, not yet the `src/` layout
above: `3DdensTransModel.py`, `PreprocessingScans.py`, `nyul_standardization.py`, plus
notebooks (`BraTS_OPtimizations.ipynb`, `DenseTrans_Training_Script_BraTS-PED.ipynb`,
`ScansVisualization.ipynb`). They have different channel counts and preprocessing
choices baked in. Do not restructure into `src/` until that's been explicitly decided
(see SETUP.md §"Day 1" and `docs/OPEN-QUESTIONS.md`) — explain the tradeoff first.
Separately, Google Drive already holds real run artifacts from prior work outside this
repo (`best_brats_v15.pt`, `best_brats_v16.pt`, `prob_maps_v16/`) — that history predates
`experiments/` and isn't reconstructable from git alone; ask before assuming what a given
checkpoint version represents.

## Non-negotiable rules
1. **No skull-stripping** anywhere in the training or inference pipeline.
2. **Patient-level splits only.** Never split by patch, slice, or augmented copy.
   The split file `configs/splits.json` is generated once and committed; do not regenerate it silently.
3. **Every experiment gets a config file and a run ID.** `python -m src.train --config configs/<name>.yaml`
   must be the only way training starts.
4. **Never report a number that was not produced by `src/metrics/evaluate.py`.**
   If you estimate, label it as an estimate.
5. **Never claim a training run succeeded unless you have read the log.** Read
   `experiments/<run_id>/train.log` or the Drive mirror before reporting results.
6. Do not delete or overwrite files under `experiments/` or the preprocessed data directory.
7. Prefer editing existing modules over creating parallel copies. No `train_v2_final_FIXED.py`.

## Machine and paths
- Local dev machine: Linux, no GPU, no local Google Drive mount. Code editing, CPU-only
  tests, and log/metric analysis only. GPU training happens on Colab.
- The Google account for Colab is **blimbasi@usc.edu** — see `configs/paths.yaml`.
- **Primary Colab bridge: the official `colab` CLI** (`google-colab-cli`, installed at
  `~/.local/bin/colab`, authenticated with `--auth oauth2`). `colab new`/`exec`/`run`
  provision and drive GPU sessions headlessly from this machine — no copy-pasting cells
  into a browser tab. `colab log` reads output, `colab download` retrieves checkpoints.
  See `.claude/agents/colab-runner.md` for the full command set.
- **Live-notebook debugging: `colab-mcp`**, a project-scoped MCP server (`.mcp.json`)
  that bridges to a Colab notebook already open in the user's browser — use when the
  human is working in a notebook by hand and wants Claude to see/react to it directly.
- **Legacy fallback: git + Drive mirror.** `scripts/colab_bootstrap.py` prints a cell to
  paste manually; results mirror to `/content/drive/MyDrive/brats_runs/<run_id>/`, read
  via the **MCP Google Drive connector** (`mcp__claude_ai_Google_Drive__*`) since this
  machine has no local Drive mount — not `scripts/tail_run.py`. Only used if the CLI is
  unavailable. See `scripts/colab_ssh_setup.md` for why the cloudflared SSH tunnel this
  originally used was retired (Colab's ToS restricts unmanaged remote shells on free
  tier and kills them without warning; the `colab` CLI's own keep-alive + `colab ssh`
  supersede it).

## Working agreements with me (the human)
- Explain before you implement anything architectural. Give me the tradeoff, not just the code.
- When a decision needs my input, append it to `docs/OPEN-QUESTIONS.md` and tell me — don't guess.
- After any substantive session, update `docs/PROGRESS.md` and `docs/DECISIONS.md`.
- I want to learn this domain, not just ship it. Flag papers/concepts I should read in
  `docs/OPEN-QUESTIONS.md` under "Reading for me".

## Delegation policy
Use subagents for anything that generates volume you won't reference again — literature
searches, log parsing, full-repo scans, ablation sweeps. Return summaries, not transcripts.
See `.claude/agents/` for the specialist roster.
