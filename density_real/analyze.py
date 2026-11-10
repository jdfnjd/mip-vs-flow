"""Summarise the Tool-Hang demo-count sweep. The analysis is fixed before seeing results.

Primary metric: p(method, N, seed), final checkpoint, official deterministic
sampling, the same 400 seeded initial states. For each method pair A-B:
  g(N, s) = 100 (p_A - p_B), paired by seed (seed s is subset s and training seed s)
  d(s) = g(25, s) - g(200, s), positive when A's advantage grows with fewer demos
Reported: all three d(s), mean, sd and mean +- 4.303 sd / sqrt(3), which is fragile
with three replicates. A bootstrap over the bank
indices (10000 resamples, same indices everywhere) shows evaluation-bank
uncertainty only, not training-seed uncertainty. All results are descriptive: no
choice between sampling modes, checkpoints or sizes is made from the outcomes.

The paper comparison at N=200 is a reference, not a pass/fail gate: the paper
averages the last five evaluations 20k steps apart, this is one final checkpoint.

Fails unless the planned grid is complete, unless --partial, which prints cells only.

Usage: python analyze.py --root ~/mip-density --tag sweep [--sizes 200,25] [--out results.jsonl]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

KEY = {"flow": "mean_success_9", "mip": "mean_success_1", "regression": "mean_success_1"}
PAIRS = (("flow", "regression"), ("mip", "regression"), ("flow", "mip"))
PAPER_CHI_UNET = {"regression": (0.59, 0.73), "mip": (0.71, 0.82), "flow": (0.73, 0.87)}  # Table 7: average, best
PADDING = 7  # an episode of length L gives L - 7 windows at horizon 16, checked in preflight


def load_final(attempt: Path, meta: dict, mode: str, episodes: int) -> dict | None:
    f = attempt / f"final_{mode}_{episodes}.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text())
    assert r["ckpt_md5"] == meta["ckpt_md5"] and r["dataset_md5"] == meta["dataset_md5"], f"{f}: other run"
    assert r["sample_mode"] == mode and r["episodes"] == episodes == len(r["episode_success"]), f"{f}: episodes"
    assert 0 <= r["success"] <= 1 and set(r["episode_success"]) <= {0.0, 1.0}, f"{f}: values"
    assert abs(np.mean(r["episode_success"]) - r["success"]) < 1e-9, f"{f}: mean"
    return r


def run_summary(attempt: Path, meta: dict, method: str, episodes: int, a) -> dict:
    assert (meta["steps"], meta["eval_freq"], meta["eval_episodes"]) == (a.steps, a.eval_freq, a.eval_episodes), \
        f"{attempt}: unplanned settings {meta}"
    key = KEY[method]
    evals = [r for r in map(json.loads, (attempt / "logs/metrics.jsonl").open()) if key in r]
    vals = [r[key] for r in evals]
    assert all(math.isfinite(v) for v in vals)
    steps = [r["step"] for r in evals]
    assert steps == [a.eval_freq * (i + 1) - 1 for i in range(a.steps // a.eval_freq)], f"{attempt}: eval steps"
    assert abs(evals[-1]["best_" + key] - max(vals)) < 1e-9, f"{attempt}: best differs from the log"
    out = {"train_evals": vals, "train_last": vals[-1], "train_mean": sum(vals) / len(vals), "attempt": attempt.name,
           "hours": meta["hours"]}
    with h5py.File(meta["dataset"], "r") as f:
        out["windows"] = int(f["data"].attrs["total"]) - PADDING * len(f["data"])
    out["passes"] = a.steps * 1024 / out["windows"]
    for mode in ("zero", "stochastic"):
        out[f"record_{mode}"] = load_final(attempt, meta, mode, episodes)
        if out[f"record_{mode}"]:
            out[f"final_{mode}"] = out[f"record_{mode}"]["success"]
    return out


def seed_table(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    p = df.pivot_table(index="seed", columns=["n_demos", "method"], values="final_zero")
    return 100 * (p.xs(a, level="method", axis=1) - p.xs(b, level="method", axis=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/mip-density")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sizes", default="200,25")
    ap.add_argument("--methods", default="flow,mip,regression")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--eval_freq", type=int, default=60_000)
    ap.add_argument("--eval_episodes", type=int, default=40)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve() / "runs" / args.tag
    sizes, seeds = [int(s) for s in args.sizes.split(",")], [int(s) for s in args.seeds.split(",")]
    methods = args.methods.split(",")

    rows, missing = [], []
    for n, method, seed in itertools.product(sizes, methods, seeds):
        cell = root / f"N{n}_{method}_s{seed}"
        done = sorted(cell.glob("attempt_*/DONE.json"))
        if not done:
            missing.append(f"{cell.name}: no completed run")
            continue
        assert len(done) == 1, f"{cell}: more than one completed attempt"
        meta = json.loads(done[0].read_text())
        assert (meta["n_demos"], meta["method"], meta["seed"]) == (n, method, seed), f"{cell}: wrong manifest"
        s = run_summary(done[0].parent, meta, method, args.episodes, args)
        if not s["record_zero"]:
            missing.append(f"{cell.name}: no primary evaluation")
        rows.append({"n_demos": n, "method": method, "seed": seed, **s})
    pd.set_option("display.width", 200)
    df = pd.DataFrame(rows)
    cols = [c for c in ("n_demos", "method", "seed", "windows", "passes", "hours", "train_last", "train_mean",
                        "final_zero", "final_stochastic") if c in df]
    if missing:
        print("INCOMPLETE, NOT THE PLANNED EXPERIMENT:\n  " + "\n  ".join(missing))
        if not args.partial:
            raise SystemExit(1)
        if rows:
            print(df[cols].round(3).to_string())
        return

    records = {(r.n_demos, r.method, r.seed): r.record_zero for r in df.itertuples()}
    first = next(iter(records.values()))
    for k, r in records.items():
        assert r["bank"] == first["bank"] and r["bank_seeds"] == first["bank_seeds"], f"{k}: other bank"
        assert r["init_hashes"] == first["init_hashes"], f"{k}: other initial states"
    assert len(set(first["bank_seeds"])) == args.episodes
    print(f"grid complete; all primary evaluations share {args.episodes} initial states "
          f"({len(set(first['init_hashes']))} distinct)")
    print(df[cols].round(3).to_string())

    if args.out:
        df.drop(columns=["record_zero", "record_stochastic"]).to_json(args.out, orient="records", lines=True)
    g = df.groupby(["n_demos", "method"])
    print("\nwindows and passes through the data at each size\n",
          df.groupby(["n_demos", "seed"])[["windows", "passes"]].first().round(0).to_string())
    print("\nprimary: final_zero, mean over seeds (%)\n", (100 * g["final_zero"].mean()).unstack("method").round(1).to_string())
    print("\nsame, sd over seeds (points)\n", (100 * g["final_zero"].std()).unstack("method").round(1).to_string())
    if "final_stochastic" in df:
        print("\nsecondary: flow with stochastic sampling (%)\n",
              (100 * df[df.method == "flow"].pivot_table(index="n_demos", columns="seed", values="final_stochastic")).round(1).to_string())
    print("\ntraining evaluations (%), 40 episodes each, not paired\n",
          (100 * g["train_last"].mean()).unstack("method").round(1).to_string())

    if 200 in sizes:
        print("\nreference at N=200, not a pass/fail gate (the protocols differ)")
        for m in methods:
            p = df[(df.n_demos == 200) & (df.method == m)]
            print(f"  {m}: final {100 * p.final_zero.mean():.1f}%, paper average {100 * PAPER_CHI_UNET[m][0]:.0f}%, "
                  f"paper best {100 * PAPER_CHI_UNET[m][1]:.0f}%")

    print("\nper seed gaps (points)")
    for a, b in PAIRS:
        if a in methods and b in methods:
            t = seed_table(df, a, b)
            print(f"{a}-{b}\n", t.round(1).to_string())
            print(" mean", t.mean().round(1).to_dict())

    rng = np.random.default_rng(0)
    idx = rng.integers(0, args.episodes, size=(10_000, args.episodes))
    S = {k: np.asarray(r["episode_success"]) for k, r in records.items()}
    boot = {(n, m): np.mean([S[(n, m, s)][idx].mean(axis=1) for s in seeds], axis=0)
            for n, m in itertools.product(sizes, methods)}
    print("\nchange of the gap with fewer demos (points)")
    for label, low in (("primary d = g25 - g200", 25), ("secondary g50 - g200", 50)):
        if low not in sizes or 200 not in sizes:
            continue
        for a, b in PAIRS:
            if a not in methods or b not in methods:
                continue
            t = seed_table(df, a, b)
            d = t[low] - t[200]
            sd = d.std()
            half = 4.303 * sd / math.sqrt(len(d))
            bd = 100 * ((boot[(low, a)] - boot[(low, b)]) - (boot[(200, a)] - boot[(200, b)]))
            print(f"{label}, {a}-{b}: per seed {d.round(1).tolist()}, mean {d.mean():.1f}, sd {sd:.1f}, "
                  f"t interval [{d.mean() - half:.1f}, {d.mean() + half:.1f}], "
                  f"bank-only 95% [{np.percentile(bd, 2.5):.1f}, {np.percentile(bd, 97.5):.1f}]")


if __name__ == "__main__":
    main()
