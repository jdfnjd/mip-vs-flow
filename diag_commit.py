"""Diagnostic: how strongly does each policy's *first* action chunk commit to a
side in the hard corner (alpha=1)? We report the lateral displacement of the
first 8-step chunk at the start state, as a fraction of the expert's
(|dx| of a demo over its first 8 steps). 0 = goes straight at the wall
(the conditional mean), 1 = commits to a gap like the expert."""
import json, sys
import numpy as np, torch
import mm_nav, run_fix
from mm_nav import TaskCfg, TrainCfg, make_dataset, train

dev = "cuda" if torch.cuda.is_available() else "cpu"
cfg = TaskCfg(alpha=1.0, n_demos=250, start_y=-0.5, approach_y=0.15, start_x_range=0.1)
rows = []
for seed in (0, 1, 2):
    obs, act, _ = make_dataset(cfg, seed)
    start = np.isclose(obs[:, 1], cfg.start_y)
    expert = np.abs(act[start].reshape(-1, cfg.horizon, 2)[:, :, 0].sum(1)).mean()
    g = np.random.default_rng(seed)
    x0 = g.uniform(-cfg.start_x_range, cfg.start_x_range, 1000).astype(np.float32)
    o = torch.as_tensor(np.stack([x0, np.full_like(x0, cfg.start_y)], 1), device=dev)
    mm_nav.LOSSES["mip0.9"] = run_fix.make_mip_loss(0.9)
    mm_nav.LOSSES["mip0.5"] = run_fix.make_mip_loss(0.5)
    for loss, variants in [("regression", ["mip@0.9:first"]), ("mip0.9", ["mip@0.9:first", "mip@0.9"]),
                           ("mip0.5", ["mip@0.5"]), ("flow", ["flow@1", "flow@2", "flow@20"])]:
        net, on, an, _ = train(loss, obs, act, TrainCfg(), seed, dev)
        for v in variants:
            for stoch in (False, True):
                torch.manual_seed(seed)
                if v.endswith(":first"):  # plain first-pass output net(o, 0, t=0)
                    if stoch: continue
                    B = o.shape[0]
                    a = net(torch.zeros(B, device=dev), torch.zeros(B, act.shape[1], device=dev), on.n(o))
                else:
                    a = run_fix.sample(net, v, on.n(o), act.shape[1], stochastic=stoch)
                a = an.u(a).detach().view(-1, cfg.horizon, 2)
                lat = a[:, :, 0].sum(1).abs().mean().item() / expert
                fwd = a[:, :, 1].sum(1).mean().item()
                name = f"{loss}:{v}{' stoch' if stoch else ''}"
                rows.append((seed, name, lat, fwd))
                print(f"s={seed} {name:32s} lateral/expert={lat:.2f} forward={fwd:.3f}", flush=True)
import pandas as pd
df = pd.DataFrame(rows, columns=["seed", "policy", "lateral", "forward"])
print(df.groupby("policy", sort=False)[["lateral", "forward"]].mean().round(3).to_string())
df.to_json("results/diag_commit.jsonl", orient="records", lines=True)
