"""Why does MIP fail in the hard corner while deterministic (z=0) flow does not?

Hard corner: alpha=1, short runway, near-symmetric starts, whole 8-step chunk
executed open-loop. Two hypotheses are tested, without touching the rest of the
pipeline:

  (1) iteration count - flow is evaluated with 1 / 2 / 5 / 20 Euler steps;
  (2) noise level     - MIP is trained with t* in {0.9, 0.5, 0.2} (paper form,
                        I_t* = t* a + (1 - t*) z) and evaluated with and without
                        noise in the second step.

Usage: python run_fix.py --seed_ids 0
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

import mm_nav
from mm_nav import TaskCfg, TrainCfg, evaluate, make_dataset, mode_stats, sqnorm, train


def make_mip_loss(t_star: float):
    def loss(net, obs, act):
        B = act.shape[0]
        s = torch.zeros(B, device=act.device)
        t = torch.full((B,), t_star, device=act.device)
        act_t = t_star * act + (1 - t_star) * torch.randn_like(act)
        p0 = net(s, torch.zeros_like(act), obs)
        p1 = net(t, act_t, obs)
        return (sqnorm(p0 - act) + sqnorm((p1 - act) / (1 - t_star))).mean()

    return loss


@torch.no_grad()
def sample(net, method, obs, act_dim, flow_steps=20, stochastic=True):
    """method: 'flow@<steps>' or 'mip@<t*>'; `stochastic` adds noise (flow: z~N,
    mip: noise in the second-step input)."""
    B, dev = obs.shape[0], obs.device
    kind, arg = method.split("@")
    zeros = torch.zeros(B, act_dim, device=dev)
    if kind == "flow":
        n = int(arg)
        x = torch.randn(B, act_dim, device=dev) if stochastic else zeros
        ts = torch.linspace(0, 1, n + 1, device=dev)
        for i in range(n):
            x = x + net(torch.full((B,), ts[i].item(), device=dev), x, obs) * (ts[i + 1] - ts[i])
        return x
    t_star = float(arg)
    a0 = net(torch.zeros(B, device=dev), zeros, obs)
    x = t_star * a0
    if stochastic:
        x = x + (1 - t_star) * torch.randn_like(x)
    return net(torch.full((B,), t_star, device=dev), x, obs)


mm_nav.sample = sample  # evaluate() looks the sampler up on the module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_ids", default="0,1,2")
    ap.add_argument("--eval_eps", type=int, default=500)
    ap.add_argument("--out", default="results/fix.jsonl")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    tcfg = TrainCfg()
    base = TaskCfg(n_demos=250, start_y=-0.5, approach_y=0.15, start_x_range=0.1)
    for seed in [int(s) for s in args.seed_ids.split(",")]:
        for alpha in (1.0, 0.0):
            cfg = replace(base, alpha=alpha)
            obs, act, _ = make_dataset(cfg, seed)
            runs = [("flow", "flow", [f"flow@{n}" for n in (1, 2, 5, 20)])]
            for ts in (0.9, 0.5, 0.2):
                mm_nav.LOSSES[f"mip{ts}"] = make_mip_loss(ts)
                runs.append((f"mip{ts}", "mip", [f"mip@{ts}"]))
            for loss_name, kind, variants in runs:
                net, on, an, _ = train(loss_name, obs, act, tcfg, seed, device)
                for variant in variants:
                    for stoch in (False, True):
                        for ex in (4, 8):
                            ecfg = replace(cfg, n_action_exec=ex)
                            res = evaluate(net, on, an, variant, ecfg, args.eval_eps, seed, device, stochastic=stoch)
                            rec = {"seed": seed, "kind": kind, "variant": variant, "stochastic": stoch,
                                   **asdict(ecfg), "success": res["success"], "crash": res["crash"],
                                   "tv": mode_stats(res, ecfg)["tv"]}
                            with out.open("a") as f:
                                f.write(json.dumps(rec) + "\n")
                            print(f"s={seed} a={alpha} ex={ex} {variant:9s} stoch={int(stoch)} "
                                  f"succ={res['success']:.3f} tv={rec['tv']:.3f}", flush=True)


if __name__ == "__main__":
    main()
