# Progress log

Newest first. Maintained by the `scribe` agent via `/wrap-up`.

## 2026-08-22 — Project scaffold + Colab bridge setup
**Done:** Copied the Claude Code scaffold (CLAUDE.md, `.claude/`, `docs/`, `scripts/`,
`configs/paths.yaml`) into the real repo and adapted it for this machine (Linux, no
local Drive mount, `blimbasi@usc.edu` as the Colab account). Added `requirements.txt`
inferred from the existing scripts' imports. Tried a cloudflared SSH tunnel to the Colab
VM; it failed (Cloudflare 1033, twice) because Colab's ToS kills unmanaged remote shells
on free tier. Replaced it with two official Google tools: `google-colab-cli` (installed,
`~/.local/bin/colab`) for headless `colab new`/`exec`/`run`/`log`/`download`, and
`colab-mcp` (added as a project-scoped MCP server in `.mcp.json`) for live debugging of
a notebook open in the browser. Updated `colab-runner` agent and the `launch-run`/
`check-run` skills to use the CLI as primary, Drive-mirror-via-MCP as fallback. See D-002.
**Results:** n/a — no training run yet.
**Broken / in flight:** `colab` CLI needs a one-time interactive OAuth login
(`colab --auth oauth2 new -s <name>`, account `blimbasi@usc.edu`) — pending on the human.
`colab-mcp` is added to `.mcp.json` but pending approval until `claude` is restarted
inside the repo. Repo is still 3 flat legacy scripts, not yet the `src/` layout in
CLAUDE.md — Day-1 consolidation assessment (5 vs 15 channels, CLAHE/gradient-magnitude
channel, 2.5D model) has not been done yet.
**Next:** finish `colab` CLI OAuth login; restart Claude Code inside
`BraTs-Challenge-6` and approve the `colab-mcp` server; then do the Day-1 read-and-report
pass on the 3 legacy scripts before any `src/` restructuring.
