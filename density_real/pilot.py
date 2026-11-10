"""Timing pilot on GPUs: sustained speed at each number of jobs per GPU, before any result exists.

Each concurrency runs short jobs of two lengths with one evaluation at the end.
The difference gives updates per second under load; the rest is one evaluation plus
start-up, with all evaluations overlapping (the worst case). Scores are not read.
Prints the projected time of the 27-run and 18-run plans, to choose before the sweep.

Usage: python pilot.py --root ~/mip-density
"""

from __future__ import annotations

import argparse
import heapq
import json
import subprocess
import sys
from pathlib import Path

GRIDS = {1: ("25", "0"), 3: ("25", "0,1,2"), 4: ("200,25", "0,1")}  # 3, 9 and 12 jobs on 3 GPUs
SHORT, LONG = 1000, 5000
METHODS = ("flow", "mip", "regression")


def makespan(hours: list[float], slots: int) -> float:
    heap = [0.0] * slots
    for h in sorted(hours, reverse=True):
        heapq.heappush(heap, heapq.heappop(heap) + h)
    return max(heap)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/mip-density")
    ap.add_argument("--gpus", default="0,1,2")
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    here = Path(__file__).resolve().parent
    for k, (sizes, seeds) in GRIDS.items():
        for steps in (SHORT, LONG):
            subprocess.check_call([sys.executable, str(here / "run_grid.py"), "--root", str(root), "--gpus", args.gpus,
                                   "--tag", f"pilot_k{k}_{steps}", "--per_gpu", str(k), "--steps", str(steps),
                                   "--eval_freq", str(steps), "--save_freq", str(steps), "--sizes", sizes,
                                   "--seeds", seeds])

    def hours(tag: str, method: str) -> float:
        h = [json.loads(p.read_text())["hours"] for p in (root / "runs" / tag).glob(f"*_{method}_*/attempt_*/DONE.json")]
        return sum(h) / len(h)

    final = {}  # upper bound for a 400-episode final evaluation: ten times the 40-episode one, with start-up
    for m in METHODS:
        short, long = hours(f"pilot_k1_{SHORT}", m), hours(f"pilot_k1_{LONG}", m)
        final[m] = 10 * (short - SHORT * (long - short) / (LONG - SHORT))
    for k in GRIDS:
        per_run = {}
        for m in METHODS:
            short, long = hours(f"pilot_k{k}_{SHORT}", m), hours(f"pilot_k{k}_{LONG}", m)
            speed = (LONG - SHORT) / (3600 * (long - short))
            other = short - SHORT / (3600 * speed)  # one evaluation plus start-up
            per_run[m] = 300_000 / (3600 * speed) + 5 * other
            print(f"k={k} {m}: {speed:.1f} updates/s, evaluation+start-up {60 * other:.1f} min, "
                  f"300k run about {per_run[m]:.1f} h")
        slots = 3 * k
        for label, reps in (("27 runs", 9), ("18 runs", 6)):
            train = makespan([per_run[m] for m in METHODS] * reps, slots)
            evals = makespan([final[m] for m in METHODS] * reps, 3)  # run_evals with one job per GPU
            print(f"k={k} {label}: training about {train:.1f} h + final evaluations at most {evals:.1f} h "
                  f"= {train + evals:.1f} h")


if __name__ == "__main__":
    main()
