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
