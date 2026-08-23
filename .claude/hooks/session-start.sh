#!/usr/bin/env bash
# Injected into Claude's context at session start (stdout of a SessionStart hook
# becomes additional context). Keep it short and factual.
set -u

echo "=== BraTS-PEDs session context ==="
echo "commit: $(git log -1 --oneline 2>/dev/null || echo 'not a git repo')"

DIRTY=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
echo "uncommitted files: ${DIRTY}"

echo "--- last 12 lines of docs/PROGRESS.md ---"
tail -n 12 docs/PROGRESS.md 2>/dev/null || echo "(no PROGRESS.md yet)"

echo "--- blocking open questions ---"
sed -n '/## Blocking me/,/## Reading for me/p' docs/OPEN-QUESTIONS.md 2>/dev/null | head -n 20 \
  || echo "(none)"

echo "--- most recent runs ---"
ls -1t experiments 2>/dev/null | head -n 5 || echo "(no experiments yet)"
exit 0
