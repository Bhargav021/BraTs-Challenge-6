---
paths:
  - "src/**/*.py"
  - "scripts/**/*.py"
---

# Rules for code under src/ and scripts/

- Type-hint public functions. Docstring anything non-obvious in one or two lines.
- No hardcoded paths. Read from `configs/paths.yaml` via `src/utils/paths.py`.
- No hardcoded hyperparameters. They come from the run config.
- Seed everything reachable: `random`, `numpy`, `torch`, `torch.cuda`, and set
  `torch.backends.cudnn.deterministic` when a run is meant to be reproducible.
- Tensor shape conventions: images `(B, C, D, H, W)`, labels `(B, D, H, W)` as int64.
  Annotate the expected shape in a comment wherever a rearrange or permute happens.
- Fail loudly. No bare `except:` that swallows a case and returns zeros — a silently
  skipped corrupt scan becomes a mystery score drop three days later.
- Print unbuffered (`python -u` or `flush=True`) so Colab log tailing works.
- If you change anything that affects numerics, run it and paste the real output.
  Do not describe what the code "should" print.
