"""Run sweeps for the controllable-multimodality study.

Each (task config, seed) trains regression / mip / flow on the same dataset and
evaluates them closed-loop; flow is evaluated with stochastic and z=0 sampling.
Results are appended as JSON lines so runs can be resumed.

Examples:
  python run_sweep.py --exp alpha_n   # ambiguity x dataset size
  python run_sweep.py --exp exec      # replanning interval
  python run_sweep.py --exp commit    # runway x start range
  python run_sweep.py --exp hard      # hard corner x replanning interval
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from mm_nav import TaskCfg, TrainCfg, evaluate, make_dataset, mode_stats, train

METHODS = ["regression", "mip", "flow"]


def experiments(name: str):
    base = TaskCfg()
    if name == "alpha_n":
        grid = [
            replace(base, alpha=a, n_demos=n)
            for a, n in itertools.product((0.0, 0.25, 0.5, 0.75, 1.0), (10, 50, 250))
        ]
        return grid, [0, 1, 2], TrainCfg()
    if name == "exec":
        # replanning interval: each trained model is evaluated at several
        # n_action_exec values (eval-only knob, no retraining)
        grid = [replace(base, alpha=a, n_demos=250) for a in (0.25, 0.5, 1.0)]
        return grid, [0, 1, 2], TrainCfg()
    if name == "commit":
        # time-to-commit: shorter runway + narrower start range
        geo = [(-1.2, 0.3), (-0.8, 0.3), (-0.5, 0.15)]
        grid = [
            replace(base, alpha=a, n_demos=250, start_y=sy, approach_y=ay, start_x_range=xr)
            for a, (sy, ay), xr in itertools.product((0.5, 1.0), geo, (0.5, 0.1))
        ]
        return grid, [0, 1, 2], TrainCfg()
    if name == "hard":
        # hardest geometry (short runway, near-symmetric starts) x replanning
        # interval; alpha=0 is the unimodal control for the same geometry
        grid = [
            replace(base, alpha=a, n_demos=250, start_y=-0.5, approach_y=0.15, start_x_range=0.1)
            for a in (0.0, 0.5, 1.0)
        ]
        return grid, [0, 1, 2], TrainCfg()
    raise ValueError(name)


EXEC_LISTS = {"exec": [1, 2, 4, 8], "hard": [1, 2, 4, 8]}


def key(cfg: TaskCfg, seed: int) -> str:
    return json.dumps({**asdict(cfg), "seed": seed}, sort_keys=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="alpha_n")
    ap.add_argument("--eval_eps", type=int, default=500)
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=None, help="override train steps")
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None, help="use only the first n seeds")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    ap.add_argument("--seed_ids", default=None, help="comma-separated seeds (for parallel shards)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.threads:
        torch.set_num_threads(args.threads)
    grid, seeds, tcfg = experiments(args.exp)
    for f in ("steps", "hidden", "depth"):
        if getattr(args, f):
            tcfg = replace(tcfg, **{f: getattr(args, f)})
    if args.seeds:
        seeds = seeds[: args.seeds]
    if args.seed_ids:
        seeds = [int(s) for s in args.seed_ids.split(",")]
    out = Path(args.out or f"results/{args.exp}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["key"], r["method"], r["n_action_exec"]))

    total = len(grid) * len(seeds)
    i = 0
    for cfg, seed in itertools.product(grid, seeds):
        i += 1
        k = key(cfg, seed)
        obs, act, modes = make_dataset(cfg, seed)
        for method in METHODS:
            evals = [("flow", True), ("flow_z0", False)] if method == "flow" else [(method, True)]
            execs = EXEC_LISTS.get(args.exp, [cfg.n_action_exec])
            if all((k, name, e) in done for name, _ in evals for e in execs):
                continue
            t0 = time.time()
            net, on, an, final_loss = train(method, obs, act, tcfg, seed, device)
            for (name, stoch), e in itertools.product(evals, execs):
                ecfg = replace(cfg, n_action_exec=e)
                res = evaluate(net, on, an, method, ecfg, args.eval_eps, seed, device, stochastic=stoch)
                ms = mode_stats(res, ecfg)
                rec = {
                    "key": k,
                    "exp": args.exp,
                    "method": name,
                    "seed": seed,
                    **{kk: v for kk, v in asdict(ecfg).items()},
                    "n_samples": int(len(obs)),
                    "success": res["success"],
                    "crash": res["crash"],
                    "timeout": res["timeout"],
                    "tv": ms["tv"],
                    "gap_hist": ms["hist"].tolist(),
                    "x0_bin_counts": ms["counts"].tolist(),
                    "final_loss": final_loss,
                    "train_sec": time.time() - t0,
                }
                with out.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(
                    f"[{i}/{total}] a={cfg.alpha} N={cfg.n_demos} K={cfg.K} D={cfg.distractor_dim} "
                    f"sy={cfg.start_y} xr={cfg.start_x_range} "
                    f"p={cfg.mode_probs} ex={e} s={seed} {name:10s} succ={res['success']:.3f} "
                    f"crash={res['crash']:.3f} tv={ms['tv']:.3f} ({time.time() - t0:.0f}s)",
                    flush=True,
                )


if __name__ == "__main__":
    main()
