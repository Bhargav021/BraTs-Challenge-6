---
paths:
  - "configs/**"
  - "experiments/**"
---

# Rules for configs and experiment records

- One config = one run = one run ID. Configs are immutable once a run has started;
  to change something, make a new config.
- `experiments/<run_id>/` is append-only. Never edit or delete a past result.
- Every config carries a `notes:` field stating what it changed relative to its parent.
- Results in markdown must state the metric explicitly: "lesion-wise Dice (ET) 0.62"
  never just "Dice 0.62". Voxel-wise numbers must be labeled as such.
