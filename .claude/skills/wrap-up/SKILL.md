---
name: wrap-up
description: End-of-session ritual — update the progress log, record decisions, list open questions and reading, and hand off cleanly. Use when I say we're done, wrap up, end of session, or before I stop for the day.
---

# Wrap up the session

Delegate to the `scribe` subagent, then verify its work yourself.

1. `scribe` updates `docs/PROGRESS.md`, `docs/DECISIONS.md`, and `docs/OPEN-QUESTIONS.md`
   from this session.
2. Check `git status`. List anything uncommitted and propose a commit message. Do not
   commit without asking.
3. If any run is still going on Colab, note its run ID and expected finish in PROGRESS.md
   so the next session starts by checking it.
4. Print a short handoff to me, in this shape and nothing longer:
   - What changed today (3 bullets max)
   - What I need to decide before the next session (with your recommendation)
   - What to read, if anything, and why it matters here
   - The exact first command for next session
