"""Mechanism diagnostics for the `density` sweep (saved checkpoints, no training).

Success and mode TV cannot tell apart (a) a policy that represents both routes
at a given observation from (b) a deterministic policy that picks a route from
the per-episode distractor values or from the exact start position. These
diagnostics roll each policy out closed loop from *fixed* initial observations
and record the realised outcome: left gap, right gap, or failure (crash or
timeout). Every metric uses realised outcomes, not first-chunk proxies.

For each checkpoint, sampler (deterministic; flow also stochastic) and
replanning interval (4, 8):

  train_match       rollouts from each training demo's exact initial
                    observation (position + distractors): P(outcome = that
                    demo's route).
  shuf_recipient    the same starts with distractors taken from another demo,
  shuf_donor        over several cyclic shifts: P(outcome = recipient's route)
  shuf_baseline     and P(outcome = donor's route); baseline = the chance that
                    a random other demo has the same route (from the labels).
                    Donor agreement well above baseline = the policy recalls
                    routes through distractors.
  two_sided_fixed   stochastic flow: K rollouts at each fixed held-out
                    observation; mean of 2 * min(P_left, P_right). 1 = both
                    routes equally often at the same observation, 0 = one route
                    (or failure). `two_sided_ref` is the expected value of the
                    estimator for a perfectly balanced sampler with K draws.
  two_sided_distr   deterministic samplers: at fixed start x0, M fresh
                    distractor vectors; mean over an x0 grid of
                    2 * min(P_left, P_right) across the draws. High = route
                    choice depends on irrelevant distractors.
  success_*         share of rollouts that reach the goal through a gap, for
                    each probe set.

Failures count in every denominator, so low agreement or low two-sidedness can
also come from failures. Probe observations are fixed per data seed and shared
by all samplers; the torch seed is set once per sampler and replanning
interval, so the probe sets draw successive parts of one noise stream.

Usage: python density_diag.py [--ckpt ckpt/density] [--out results/density_diag.jsonl]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import mm_nav
from mm_nav import (GOAL, GOAL_TOL, MAX_DISTRACT, MAX_STEPS, Normalizer, PolicyNet, TaskCfg, gap_centers,
                    make_dataset, segment_hits_wall)

K = 32  # rollouts per fixed observation (stochastic flow)
K_SHUF = 8  # rollouts per shuffled observation (stochastic flow)
M = 16  # fresh distractor draws per x0 (deterministic samplers)
X0_GRID = 21
N_HELDOUT = 64
N_SHIFTS = 9  # cyclic shifts used for the distractor shuffle (all if fewer exist)


def load(path: Path, device: str):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = TaskCfg(**ck["cfg"])
    net = PolicyNet(2 + cfg.distractor_dim, 2 * cfg.horizon, ck["tcfg"]["hidden"], ck["tcfg"]["depth"]).to(device)
    net.load_state_dict(ck["state"])
    net.eval()
    on = Normalizer(*[t.to(device) for t in ck["on"]])
    an = Normalizer(*[t.to(device) for t in ck["an"]])
    return net, on, an, cfg, ck["method"], ck["seed"]


@torch.no_grad()
def rollout(net, on, an, method, cfg: TaskCfg, obs0: torch.Tensor, n_exec: int, stochastic: bool):
    """Closed-loop rollouts from fixed initial observations (rows of obs0).
    Same dynamics and wall check as mm_nav.evaluate. Returns outcome per row:
    0 = left gap, 1 = right gap, -1 = failure (crash or timeout)."""
    n, dev = obs0.shape[0], obs0.device
    centers = gap_centers(cfg.K)
    c = torch.as_tensor(centers, device=dev)
    pos, distract = obs0[:, :2].clone(), obs0[:, 2:]
    goal = torch.as_tensor(GOAL, device=dev)
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    success = torch.zeros_like(alive)
    cross_x = torch.full((n,), float("nan"), device=dev)
    act_dim = 2 * cfg.horizon
    steps = 0
    while steps < MAX_STEPS and alive.any():
        a = an.u(mm_nav.sample(net, method, on.n(torch.cat([pos, distract], 1)), act_dim, 20, stochastic))
        a = a.view(n, cfg.horizon, 2)
        for j in range(n_exec):
            new = pos + a[:, j] * alive[:, None]
            blocked = alive & segment_hits_wall(pos, new, centers)
            y0, y1 = pos[:, 1], new[:, 1]
            crosses = alive & (torch.sign(y0) != torch.sign(y1)) & (y0 != y1)
            frac = (-y0 / (y1 - y0 + 1e-12)).clamp(0, 1)
            xc = pos[:, 0] + frac * (new[:, 0] - pos[:, 0])
            cross_x = torch.where(crosses & torch.isnan(cross_x), xc, cross_x)
            pos = torch.where(blocked[:, None], pos, new)
            reached = alive & ~blocked & ((pos - goal).norm(dim=1) < GOAL_TOL)
            success |= reached
            alive &= ~(blocked | reached)
            steps += 1
    out = torch.full((n,), -1, dtype=torch.long, device=dev)
    ok = success & ~torch.isnan(cross_x)
    out[ok] = torch.argmin((cross_x[ok, None] - c).abs(), 1)
    return out


def two_sided(outcomes: torch.Tensor) -> float:
    """outcomes: (groups, draws). Mean over groups of 2 * min(P_left, P_right)."""
    pl, pr = (outcomes == 0).float().mean(1), (outcomes == 1).float().mean(1)
    return float((2 * torch.minimum(pl, pr)).mean())


def two_sided_ref(k: int) -> float:
    """E[2 * min(X, k - X) / k] for X ~ Binomial(k, 1/2): the estimator's value for a balanced sampler."""
    return sum(math.comb(k, x) * 0.5 ** k * 2 * min(x, k - x) / k for x in range(k + 1))


def analyse(path: Path, device: str):
    net, on, an, cfg, method, seed = load(path, device)
    obs, _, modes = make_dataset(cfg, seed)
    starts = np.flatnonzero(np.r_[True, np.diff(obs[:, 1]) < -0.5])  # first row of each demo
    assert len(starts) == cfg.n_demos
    o_tr = torch.as_tensor(obs[starts], device=device)
    route = torch.as_tensor(modes, device=device)  # 0 = left gap (x = -0.6), 1 = right gap
    n = cfg.n_demos

    # fixed probe observations, shared by every sampler of this data seed (prefix-paired across D)
    g = np.random.default_rng([seed, 2])
    shifts = sorted(g.choice(np.arange(1, n), size=min(N_SHIFTS, n - 1), replace=False).tolist())
    x_ho = g.uniform(-cfg.start_x_range, cfg.start_x_range, N_HELDOUT)
    d_ho = g.standard_normal((N_HELDOUT, MAX_DISTRACT))[:, : cfg.distractor_dim]
    o_ho = np.concatenate([x_ho[:, None], np.full((N_HELDOUT, 1), cfg.start_y), d_ho], 1)
    xs = np.linspace(-cfg.start_x_range, cfg.start_x_range, X0_GRID)
    d_b = g.standard_normal((X0_GRID * M, MAX_DISTRACT))[:, : cfg.distractor_dim]
    o_b = np.concatenate([np.repeat(xs, M)[:, None], np.full((X0_GRID * M, 1), cfg.start_y), d_b], 1)
    o_ho, o_b = (torch.as_tensor(a, dtype=torch.float32, device=device) for a in (o_ho, o_b))

    lab = modes
    n_l, n_r = int((lab == 0).sum()), int((lab == 1).sum())
    baseline = (n_l * (n_l - 1) + n_r * (n_r - 1)) / (n * (n - 1))

    variants = [(method, False)] + ([("flow", True)] if method == "flow" else [])
    rows = []
    for m, stoch in variants:
        name = {"regression": "regression", "mip": "mip"}.get(m, "flow" if stoch else "flow_z0")
        for n_exec in (4, 8):
            torch.manual_seed(1000 * seed + n_exec)
            k = K if stoch else 1
            out_tr = rollout(net, on, an, m, cfg, o_tr.repeat(k, 1), n_exec, stoch).view(k, n)
            rec = {"alpha": cfg.alpha, "n_demos": n, "distractor_dim": cfg.distractor_dim, "seed": seed,
                   "policy": name, "n_action_exec": n_exec,
                   "train_match": float((out_tr == route[None]).float().mean()),
                   "success_train": float((out_tr >= 0).float().mean())}
            if cfg.distractor_dim > 0:
                ks = K_SHUF if stoch else 1
                rec_m, don_m = [], []
                for s in shifts:
                    donor = torch.roll(torch.arange(n, device=device), s)
                    o_sh = o_tr.clone()
                    o_sh[:, 2:] = o_tr[donor, 2:]
                    out = rollout(net, on, an, m, cfg, o_sh.repeat(ks, 1), n_exec, stoch).view(ks, n)
                    rec_m.append(float((out == route[None]).float().mean()))
                    don_m.append(float((out == route[donor][None]).float().mean()))
                rec.update(shuf_recipient=float(np.mean(rec_m)), shuf_donor=float(np.mean(don_m)),
                           shuf_baseline=baseline, n_shifts=len(shifts))
            if stoch:
                out_ho = rollout(net, on, an, m, cfg, o_ho.repeat(K, 1), n_exec, True).view(K, N_HELDOUT).T
                rec.update(two_sided_fixed=two_sided(out_ho), two_sided_ref=two_sided_ref(K),
                           success_heldout=float((out_ho >= 0).float().mean()))
            else:
                out_b = rollout(net, on, an, m, cfg, o_b, n_exec, False).view(X0_GRID, M)
                rec.update(two_sided_distr=two_sided(out_b) if cfg.distractor_dim > 0 else 0.0,
                           two_sided_ref=two_sided_ref(M), success_heldout=float((out_b >= 0).float().mean()))
            rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/density")
    ap.add_argument("--out", default="results/density_diag.jsonl")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for p in sorted(Path(args.ckpt).glob("*.pt")):
        rows += analyse(p, device)
        print(p.name, flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(len(rows), "rows ->", out)


if __name__ == "__main__":
    main()
