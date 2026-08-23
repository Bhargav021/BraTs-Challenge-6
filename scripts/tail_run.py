#!/usr/bin/env python3
"""Read a Colab run's mirrored output from Google Drive and summarize it.

This is the channel Claude Code uses to "see" the CLI output of a job running on a
Colab GPU: Colab writes train.log and metrics.jsonl into the Drive mirror, Google
Drive syncs them to this machine, and this script reads them locally.

Usage:
    python scripts/tail_run.py --run 20260822-baseline --lines 80
    python scripts/tail_run.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

DEFAULT_MIRRORS = [
    os.environ.get("BRATS_DRIVE_ROOT", ""),
    r"G:\My Drive\brats_runs",
    str(Path.home() / "Google Drive" / "My Drive" / "brats_runs"),
    "/content/drive/MyDrive/brats_runs",
]

ALERT = re.compile(
    r"Traceback|Error|ERROR|CUDA out of memory|OutOfMemoryError|nan|NaN|WARN|Killed",
)


def find_mirror() -> Path:
    for cand in DEFAULT_MIRRORS:
        if cand and Path(cand).is_dir():
            return Path(cand)
    sys.exit(
        "No Drive mirror found. Set BRATS_DRIVE_ROOT to the folder that holds "
        "brats_runs/, or check that Google Drive is mounted."
    )


def human_age(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m ago"
    return f"{seconds / 3600:.1f}h ago"


def summarize_metrics(path: Path) -> None:
    if not path.exists():
        print("metrics.jsonl: not written yet")
        return

    rows = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # partially-synced final line

    if not rows:
        print("metrics.jsonl: empty")
        return

    key = next(
        (k for k in ("val_dice_mean", "val_dice", "val_metric") if k in rows[-1]), None
    )
    print(f"epochs recorded: {len(rows)}")

    if key:
        best = max(rows, key=lambda r: r.get(key) or -1)
        print(f"best {key}: {best.get(key):.4f} at epoch {best.get('epoch')}")
        tail = [r for r in rows[-5:] if r.get(key) is not None]
        trend = " -> ".join(f"{r[key]:.4f}" for r in tail)
        print(f"last 5 {key}: {trend}")
        if len(tail) >= 3:
            delta = tail[-1][key] - tail[0][key]
            verdict = "improving" if delta > 1e-3 else "flat" if delta > -1e-3 else "declining"
            print(f"trend: {verdict} ({delta:+.4f} over last {len(tail)} recorded epochs)")

    last = rows[-1]
    per_class = {k: v for k, v in last.items() if k.startswith(("dice_", "val_dice_"))}
    if per_class:
        print("last epoch per-class: " + ", ".join(f"{k}={v:.4f}" for k, v in per_class.items()))

    secs = last.get("epoch_seconds")
    total = last.get("total_epochs")
    if secs and total and last.get("epoch"):
        remaining = (total - last["epoch"]) * secs
        print(f"epoch time: {secs:.0f}s | projected remaining: {remaining / 3600:.1f}h")

    if "peak_vram_gb" in last:
        print(f"peak VRAM: {last['peak_vram_gb']:.1f} GB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run id, e.g. 20260822-baseline")
    ap.add_argument("--lines", type=int, default=60)
    ap.add_argument("--list", action="store_true", help="list runs in the mirror")
    args = ap.parse_args()

    mirror = find_mirror()

    if args.list or not args.run:
        runs = sorted(
            (p for p in mirror.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        print(f"mirror: {mirror}")
        for p in runs[:20]:
            print(f"  {p.name:<32} updated {human_age(time.time() - p.stat().st_mtime)}")
        return

    run_dir = mirror / args.run
    if not run_dir.is_dir():
        sys.exit(f"No such run in mirror: {run_dir}")

    log = run_dir / "train.log"
    print(f"=== run {args.run} ===")
    print(f"mirror path: {run_dir}")

    if log.exists():
        age = time.time() - log.stat().st_mtime
        print(f"train.log last written: {human_age(age)}")
        if age > 900:
            print("NOTE: no new output in >15 min. Session may have disconnected, "
                  "been preempted, or Drive sync may be lagging. Re-check before "
                  "declaring the run dead.")
    else:
        print("train.log: not present yet")

    print("\n--- metrics ---")
    summarize_metrics(run_dir / "metrics.jsonl")

    if log.exists():
        text = log.read_text(errors="replace").splitlines()
        alerts = [ln for ln in text if ALERT.search(ln)]
        if alerts:
            print(f"\n--- {len(alerts)} alert lines (last 15) ---")
            for ln in alerts[-15:]:
                print(ln)
        print(f"\n--- last {args.lines} log lines ---")
        for ln in text[-args.lines:]:
            print(ln)


if __name__ == "__main__":
    main()
