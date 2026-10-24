"""Follow-up checks suggested by Chaoyi Pan (Oct 2026).

1. Original-form MIP (I_t* = t* a + (1 - t*) z) over a finer t* grid, in two
   variants that differ only in how the first-step output is fed to step two:
     paper - step 1 predicts a, step 2 input is t* * a_hat        (paper eq.)
     repo  - step 1 predicts t* a, step 2 input is pred0 * t*      (mip_origin_loss /
             mip_origin_sampler in the official repo, i.e. about t*^2 a)
   Hard corner, alpha = 1, 8 steps per replan; alpha = 0 control included.

2. Signed sideways displacement of the first action chunk vs the initial
   horizontal position, to tell a sharp route-selection rule (sign follows x0)
   from a fixed preference for one side.

Usage: python followup.py sweep --seed_ids 0     (shard by seed; then concatenate
                                                  results/followup_tstar_s*.jsonl
                                                  into results/followup_tstar.jsonl)
       python followup.py chunk
       python followup.py plot
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

import mm_nav
import run_fix
from mm_nav import TaskCfg, TrainCfg, evaluate, make_dataset, mode_stats, sqnorm, train

HARD = TaskCfg(n_demos=250, start_y=-0.5, approach_y=0.15, start_x_range=0.1, n_action_exec=8)
T_STARS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def make_repo_origin_loss(t_star: float):
    """mip_origin_loss: step 1 regresses t* a, both terms scaled as in the repo."""
    def loss(net, obs, act):
        B = act.shape[0]
        s = torch.zeros(B, device=act.device)
        t = torch.full((B,), t_star, device=act.device)
        act_t = t_star * act + (1 - t_star) * torch.randn_like(act)
        p0 = net(s, torch.zeros_like(act), obs)
        p1 = net(t, act_t, obs)
        return (sqnorm((p0 - t_star * act) / t_star) + sqnorm((p1 - act) / (1 - t_star))).mean()

    return loss


@torch.no_grad()
def sample(net, method, obs, act_dim, flow_steps=20, stochastic=True):
    """'paper@t' / 'repo@t' for the two MIP variants, anything else -> run_fix."""
    kind, arg = method.split("@")
    if kind not in ("paper", "repo"):
        return run_fix.sample(net, method, obs, act_dim, flow_steps, stochastic)
    t_star = float(arg)
    B, dev = obs.shape[0], obs.device
    p0 = net(torch.zeros(B, device=dev), torch.zeros(B, act_dim, device=dev), obs)
    x = t_star * p0  # paper: p0 ~ a -> t* a ; repo: p0 ~ t* a -> t*^2 a
    if stochastic:
        x = x + (1 - t_star) * torch.randn_like(x)
    return net(torch.full((B,), t_star, device=dev), x, obs)


mm_nav.sample = sample


def sweep(seeds, out: Path, eval_eps: int):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for seed in seeds:
        for alpha in (1.0, 0.0):
            cfg = replace(HARD, alpha=alpha)
            obs, act, _ = make_dataset(cfg, seed)
            for ts in T_STARS:
                mm_nav.LOSSES[f"paper{ts}"] = run_fix.make_mip_loss(ts)
                mm_nav.LOSSES[f"repo{ts}"] = make_repo_origin_loss(ts)
                for form in ("paper", "repo"):
                    net, on, an, _ = train(f"{form}{ts}", obs, act, TrainCfg(), seed, dev)
                    for stoch in (False, True):
                        res = evaluate(net, on, an, f"{form}@{ts}", cfg, eval_eps, seed, dev, stochastic=stoch)
                        rec = {"seed": seed, "form": form, "t_star": ts, "stochastic": stoch, **asdict(cfg),
                               "success": res["success"], "crash": res["crash"], "tv": mode_stats(res, cfg)["tv"]}
                        with out.open("a") as f:
                            f.write(json.dumps(rec) + "\n")
                        print(f"s={seed} a={alpha} {form:5s} t*={ts} stoch={int(stoch)} "
                              f"succ={res['success']:.3f}", flush=True)


def chunk(out: Path):
    """First-chunk signed lateral displacement vs x0 (fraction of the expert's)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = replace(HARD, alpha=1.0)
    x0 = np.linspace(-cfg.start_x_range, cfg.start_x_range, 201).astype(np.float32)
    rows = []
    mm_nav.LOSSES["paper0.9"] = run_fix.make_mip_loss(0.9)
    for seed in (0, 1, 2):
        obs, act, _ = make_dataset(cfg, seed)
        start = np.isclose(obs[:, 1], cfg.start_y)
        expert = np.abs(act[start].reshape(-1, cfg.horizon, 2)[:, :, 0].sum(1)).mean()
        o = torch.as_tensor(np.stack([x0, np.full_like(x0, cfg.start_y)], 1), device=dev)
        for loss, variants in [("regression", [("regression", "mip@0.9:first", False)]),
                               ("paper0.9", [("MIP (t*=0.9)", "paper@0.9", False)]),
                               ("flow", [("flow, 1 step, z=0", "flow@1", False),
                                         ("flow, 20 steps, z=0", "flow@20", False),
                                         ("flow, 20 steps, z~N", "flow@20", True)])]:
            net, on, an, _ = train(loss, obs, act, TrainCfg(), seed, dev)
            for name, v, stoch in variants:
                torch.manual_seed(seed)
                if v.endswith(":first"):
                    B = o.shape[0]
                    a = net(torch.zeros(B, device=dev), torch.zeros(B, act.shape[1], device=dev), on.n(o))
                else:
                    a = sample(net, v, on.n(o), act.shape[1], stochastic=stoch)
                dx = an.u(a).view(-1, cfg.horizon, 2)[:, :, 0].sum(1).cpu().numpy() / expert
                rows += [{"seed": seed, "policy": name, "x0": float(x), "dx": float(d)} for x, d in zip(x0, dx)]
            print(f"seed {seed} done", flush=True)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def plot():
    import matplotlib.pyplot as plt
    import pandas as pd

    colors = {"regression": "#8c8c8c", "MIP (t*=0.9)": "#2a6fdb", "flow, 1 step, z=0": "#e3a21a",
              "flow, 20 steps, z=0": "#7a3fb0", "flow, 20 steps, z~N": "#c2372e"}
    c = pd.read_json("results/followup_chunk.jsonl", lines=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (s, g) in zip(axes, c.groupby("seed")):
        for p, h in g.groupby("policy", sort=False):
            ax.plot(h.x0, h.dx, "." if "z~N" in p else "-", color=colors[p], label=p, ms=3, lw=2)
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(0, color="k", lw=0.6)
        for y in (1, -1):
            ax.axhline(y, color="k", ls=":", lw=0.8)
        ax.set_title(f"seed {s}")
        ax.set_xlabel("initial horizontal position x0")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("signed sideways displacement of first chunk\n(fraction of expert; ±1 = commits to a gap)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Hard corner, alpha = 1: what the first action chunk does at each start position")
    fig.tight_layout()
    fig.savefig("figures/followup_chunk.png", dpi=150)

    d = pd.read_json("results/followup_tstar.jsonl", lines=True)
    d = d[d.alpha == 1]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    col = {"paper": "#2a6fdb", "repo": "#d9822b"}
    for (form, stoch), g in d.groupby(["form", "stochastic"]):
        m = g.groupby("t_star").success.agg(["mean", "std"])
        label = ("paper form (t*·â)" if form == "paper" else "repo mip_origin (pred0·t*)") + (", noise" if stoch else ", deterministic")
        ax.errorbar(m.index, 100 * m["mean"], yerr=100 * m["std"], ls="--" if stoch else "-",
                    marker="o" if form == "paper" else "s", color=col[form], capsize=3, label=label)
    ax.axhline(99.3, color="#e3a21a", lw=1.5, label="flow, 2 Euler steps, z=0 (99%)")
    ax.set_xlabel("t*")
    ax.set_ylabel("success (%)")
    ax.set_ylim(0, 102)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("Original-form MIP vs t* (hard corner, alpha = 1, 8 steps per replan)", fontsize=9)
    fig.tight_layout()
    fig.savefig("figures/followup_tstar.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["sweep", "chunk", "plot"])
    ap.add_argument("--seed_ids", default="0,1,2")
    ap.add_argument("--eval_eps", type=int, default=500)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    Path("results").mkdir(exist_ok=True)
    if args.what == "plot":
        plot()
    elif args.what == "sweep":
        seeds = [int(s) for s in args.seed_ids.split(",")]
        sweep(seeds, Path(f"results/followup_tstar_s{'_'.join(map(str, seeds))}.jsonl"), args.eval_eps)
    else:
        chunk(Path("results/followup_chunk.jsonl"))
