# Decision record

One entry per real decision. Never edit an accepted entry — supersede it.

## D-001: Evaluate with the official lesion-wise metrics from day one (YYYY-MM-DD)
Status: accepted
Context: BraTS-PEDs 2025 scores lesion-wise Dice and NSD over six regions. Voxel-wise
Dice on a validation split systematically overstates performance, especially for
multifocal disease, and would make every downstream comparison misleading.
Options: (a) start with voxel Dice for speed, swap later; (b) build the official
harness first.
Decision: (b). `src/metrics/evaluate.py` is the only source of reported numbers.
Consequences: slower start; every result is comparable to the leaderboard from run 1.
Evidence: challenge evaluation description.

## D-002: Colab bridge uses the official `colab` CLI + `colab-mcp`, not a manual SSH tunnel (2026-08-22)
Status: accepted
Context: initial setup used a cloudflared quick tunnel (`scripts/colab_ssh_setup.md`) so
Claude could get a shell on the Colab VM. It failed in practice — Cloudflare error 1033
within minutes of setup, on two separate tunnel attempts. Root cause: Colab's ToS
disallows unmanaged remote-control shells on free tier and kills them without warning.
Options: (a) keep debugging the cloudflared tunnel; (b) adopt Google's official
`google-colab-cli` (headless `colab new`/`exec`/`run`/`log`/`download`, built-in
keep-alive) and `colab-mcp` (MCP server bridging to a notebook already open in the
browser, lists Claude Code as a supported client) — both shipped June 2026.
Decision: (b), both tools. `colab` CLI is the primary path for launching/polling runs
from this machine; `colab-mcp` is for live debugging of a notebook the human is running
by hand. The git + Drive-mirror + manual-cell-paste bridge is kept as a documented
fallback only (`.claude/agents/colab-runner.md`, `scripts/colab_bootstrap.py`).
Consequences: one-time interactive OAuth login required for the `colab` CLI
(`--auth oauth2`, account `blimbasi@usc.edu`); GPU tier availability still depends on
the account's Colab subscription/compute-unit balance (`colab pay`).
Evidence: `scripts/colab_ssh_setup.md` superseded-notice; googlecolab/google-colab-cli
and googlecolab/colab-mcp READMEs on GitHub.
