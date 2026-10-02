"""Controllable-multimodality navigation benchmark for probing
"Much Ado About Noising" (Pan et al., ICLR 2026, arXiv:2512.01809).

Task: a 2D point agent starts below a wall and must reach a goal above it.
The wall has K gaps. Expert demos pass through one gap. We control:

  alpha   - mode ambiguity. With prob (1 - alpha) the expert's gap is a
            deterministic function of the observation (nearest gap to the
            start x); with prob alpha it is sampled from `mode_probs`.
            alpha=0 -> unimodal (but discontinuous) policy,
            alpha=1 -> fully multimodal policy.
  n_demos - dataset size (data density).
  K       - number of modes (gaps).
  start_y, start_x_range, approach_y - "time to commit". A short runway and a
            narrow (near-symmetric) start range force the mode decision to be
            made immediately, before closed-loop drift can break the tie.
  distractor_dim - extra per-episode constant observation dims that are
            pure noise, uncorrelated with the mode. Makes every demo's
            states unique (the "sparse data in high dimension" regime the
            paper argues hides multimodality).

Policies (objectives mirror the official repo mip/losses.py, mip/samplers.py):
  regression (RCP), mip (Minimal Iterative Policy, t*=0.9), flow (linear
  stochastic interpolant, Euler ODE). Flow is evaluated with stochastic
  (z ~ N(0,I)) and deterministic (z = 0) sampling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------------------------------------
# Environment geometry
# ----------------------------------------------------------------------------
START_Y = -1.2
GOAL = np.array([0.0, 1.2], dtype=np.float32)
WALL_HALF_THICK = 0.05
GAP_HALF_WIDTH = 0.15
STEP = 0.05  # expert speed per env step
GOAL_TOL = 0.1
MAX_STEPS = 160


def gap_centers(K: int) -> np.ndarray:
    """Gap centers, chosen so that the mean of any two symmetric gaps is wall."""
    if K == 2:
        return np.array([-0.6, 0.6], dtype=np.float32)
    # evenly spaced, symmetric, spacing 0.7
    return ((np.arange(K) - (K - 1) / 2) * 0.7).astype(np.float32)


def in_wall(x: np.ndarray | torch.Tensor, centers) -> np.ndarray | torch.Tensor:
    """x: (..., ) horizontal position at the wall crossing. True if blocked."""
    if isinstance(x, torch.Tensor):
        c = torch.as_tensor(centers, device=x.device)
        return (torch.abs(x[..., None] - c) > GAP_HALF_WIDTH).all(-1)
    return (np.abs(x[..., None] - centers) > GAP_HALF_WIDTH).all(-1)


def segment_hits_wall(p0: torch.Tensor, p1: torch.Tensor, centers) -> torch.Tensor:
    """True if the move p0 -> p1 passes through wall material.

    The wall occupies |y| <= WALL_HALF_THICK except inside the gaps. The part of
    the segment that lies within that band must stay inside a single gap; since
    a gap is an interval in x, it is enough to test the two ends of that part.
    """
    W = WALL_HALF_THICK
    y0, dy = p0[:, 1], p1[:, 1] - p0[:, 1]
    flat = dy.abs() < 1e-9
    safe = torch.where(flat, torch.ones_like(dy), dy)
    ta, tb = (-W - y0) / safe, (W - y0) / safe
    lo = torch.where(flat, torch.zeros_like(dy), torch.minimum(ta, tb).clamp(0, 1))
    hi = torch.where(flat, torch.ones_like(dy), torch.maximum(ta, tb).clamp(0, 1))
    in_band = torch.where(flat, y0.abs() < W, hi > lo)
    dx = p1[:, 0] - p0[:, 0]
    xa, xb = p0[:, 0] + lo * dx, p0[:, 0] + hi * dx
    c = torch.as_tensor(centers, device=p0.device)
    same_gap = ((xa[:, None] - c).abs() <= GAP_HALF_WIDTH) & ((xb[:, None] - c).abs() <= GAP_HALF_WIDTH)
    return in_band & ~same_gap.any(1)


# ----------------------------------------------------------------------------
# Expert data
# ----------------------------------------------------------------------------
def _polyline_rollout(pts: np.ndarray, step: float) -> np.ndarray:
    """Constant-speed positions along a polyline, starting at pts[0]."""
    seg = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    s = np.arange(0.0, total, step)
    s = np.append(s, total)
    out = np.empty((len(s), 2), dtype=np.float32)
    idx = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1)
    frac = (s - cum[idx]) / np.maximum(seg_len[idx], 1e-8)
    out[:] = pts[idx] + frac[:, None] * seg[idx]
    return out


@dataclass
class TaskCfg:
    K: int = 2
    alpha: float = 1.0
    mode_probs: list[float] | None = None  # None -> uniform
    n_demos: int = 100
    start_x_range: float = 0.5
    distractor_dim: int = 0
    horizon: int = 8  # predicted action chunk length
    n_action_exec: int = 4  # executed per replanning step (receding horizon)
    jitter: float = 0.04  # lateral jitter of expert waypoints
    start_y: float = START_Y  # runway: distance from start to the wall
    approach_y: float = 0.3  # expert lines up with the gap this far before the wall


def sample_mode(rng, x0: float, cfg: TaskCfg, centers) -> int:
    if rng.random() < cfg.alpha:
        p = cfg.mode_probs or [1.0 / cfg.K] * cfg.K
        return int(rng.choice(cfg.K, p=p))
    return int(np.argmin(np.abs(centers - x0)))


def make_dataset(cfg: TaskCfg, seed: int):
    """Returns obs (N, obs_dim), act (N, H*2), plus per-demo modes."""
    rng = np.random.default_rng(seed)
    centers = gap_centers(cfg.K)
    obs_list, act_list, modes = [], [], []
    for _ in range(cfg.n_demos):
        x0 = rng.uniform(-cfg.start_x_range, cfg.start_x_range)
        k = sample_mode(rng, x0, cfg, centers)
        modes.append(k)
        cx = centers[k] + rng.uniform(-cfg.jitter, cfg.jitter)
        pts = np.array(
            [
                [x0, cfg.start_y],
                [cx, -cfg.approach_y + rng.uniform(-0.05, 0.05)],
                [cx, cfg.approach_y + rng.uniform(-0.05, 0.05)],
                GOAL,
            ],
            dtype=np.float32,
        )
        traj = _polyline_rollout(pts, STEP)  # (T, 2)
        # pad with the goal so late chunks are "stay"
        traj = np.concatenate([traj, np.repeat(traj[-1:], cfg.horizon, 0)], 0)
        deltas = np.diff(traj, axis=0)  # (T-1, 2)
        distract = rng.standard_normal(cfg.distractor_dim).astype(np.float32)
        T = len(traj) - cfg.horizon
        for t in range(T):
            o = np.concatenate([traj[t], distract])
            a = deltas[t : t + cfg.horizon].reshape(-1)
            obs_list.append(o)
            act_list.append(a)
    obs = np.stack(obs_list).astype(np.float32)
    act = np.stack(act_list).astype(np.float32)
    return obs, act, np.array(modes)


# ----------------------------------------------------------------------------
# Network (MLP with FiLM-style time + noisy-action conditioning)
# ----------------------------------------------------------------------------
class TimeEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(1000.0) * torch.arange(half, device=t.device) / half
        )
        ang = t[:, None] * freqs[None] * 100.0
        return torch.cat([ang.sin(), ang.cos()], -1)


class ResBlock(nn.Module):
    def __init__(self, h: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.Mish(), nn.Linear(h, h))

    def forward(self, x):
        return x + self.net(x)


class PolicyNet(nn.Module):
    """f(x_t, t, obs) -> R^{act_dim}. For flow it is the velocity; for
    regression / MIP it is the action prediction (as in the official repo,
    all methods share the same `get_velocity(t, x, obs)` interface)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256, depth: int = 4):
        super().__init__()
        self.temb = TimeEmb(64)
        self.inp = nn.Linear(obs_dim + act_dim + 64, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, act_dim))

    def forward(self, t, x, obs):
        h = self.inp(torch.cat([x, obs, self.temb(t)], -1))
        return self.out(self.blocks(h))


# ----------------------------------------------------------------------------
# Losses & samplers (match official repo)
# ----------------------------------------------------------------------------
T_TWO_STEP = 0.9


def sqnorm(x):
    return (x * x).sum(-1)


def loss_regression(net, obs, act):
    B = act.shape[0]
    t = torch.zeros(B, device=act.device)
    pred = net(t, torch.zeros_like(act), obs)
    return sqnorm(pred - act).mean()


def loss_mip(net, obs, act):
    B = act.shape[0]
    s = torch.zeros(B, device=act.device)
    t = torch.full((B,), T_TWO_STEP, device=act.device)
    act_t = act + (1 - T_TWO_STEP) * torch.randn_like(act)
    p0 = net(s, torch.zeros_like(act), obs)
    p1 = net(t, act_t, obs)
    return (sqnorm((p0 - act) / T_TWO_STEP) + sqnorm((p1 - act) / (1 - T_TWO_STEP))).mean()


def loss_flow(net, obs, act):
    B = act.shape[0]
    t = torch.rand(B, device=act.device)
    z = torch.randn_like(act)
    x_t = (1 - t)[:, None] * z + t[:, None] * act
    v = net(t, x_t, obs)
    return sqnorm(v - (act - z)).mean()


LOSSES = {"regression": loss_regression, "mip": loss_mip, "flow": loss_flow}


@torch.no_grad()
def sample(net, method: str, obs, act_dim: int, flow_steps: int = 20, stochastic: bool = True):
    B = obs.shape[0]
    dev = obs.device
    zeros = torch.zeros(B, act_dim, device=dev)
    if method == "regression":
        return net(torch.zeros(B, device=dev), zeros, obs)
    if method == "mip":
        a0 = net(torch.zeros(B, device=dev), zeros, obs)
        return net(torch.full((B,), T_TWO_STEP, device=dev), a0, obs)
    if method == "flow":
        x = torch.randn(B, act_dim, device=dev) if stochastic else zeros
        ts = torch.linspace(0, 1, flow_steps + 1, device=dev)
        for i in range(flow_steps):
            s = torch.full((B,), ts[i].item(), device=dev)
            x = x + net(s, x, obs) * (ts[i + 1] - ts[i])
        return x
    raise ValueError(method)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
@dataclass
class Normalizer:
    mean: torch.Tensor
    std: torch.Tensor

    def n(self, x):
        return (x - self.mean) / self.std

    def u(self, x):
        return x * self.std + self.mean


def fit_norm(x: torch.Tensor) -> Normalizer:
    return Normalizer(x.mean(0), x.std(0).clamp_min(1e-4))


@dataclass
class TrainCfg:
    steps: int = 6000
    batch: int = 512
    lr: float = 3e-4
    hidden: int = 256
    depth: int = 4
    ema: float = 0.999


def train(method: str, obs_np, act_np, tcfg: TrainCfg, seed: int, device: str):
    torch.manual_seed(seed)
    obs = torch.as_tensor(obs_np, device=device)
    act = torch.as_tensor(act_np, device=device)
    on, an = fit_norm(obs), fit_norm(act)
    obs_n, act_n = on.n(obs), an.n(act)
    net = PolicyNet(obs.shape[1], act.shape[1], tcfg.hidden, tcfg.depth).to(device)
    ema = PolicyNet(obs.shape[1], act.shape[1], tcfg.hidden, tcfg.depth).to(device)
    ema.load_state_dict(net.state_dict())
    ema.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=tcfg.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, tcfg.steps)
    loss_fn = LOSSES[method]
    N = obs.shape[0]
    for step in range(tcfg.steps):
        idx = torch.randint(0, N, (tcfg.batch,), device=device)
        loss = loss_fn(net, obs_n[idx], act_n[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), net.parameters()):
                pe.lerp_(p, 1 - tcfg.ema)
    ema.eval()
    return ema, on, an, float(loss.item())


# ----------------------------------------------------------------------------
# Closed-loop evaluation (vectorised over episodes)
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(net, on, an, method, cfg: TaskCfg, n_eps: int, seed: int, device: str,
             stochastic: bool = True, flow_steps: int = 20):
    g = np.random.default_rng(10_000 + seed)
    centers = gap_centers(cfg.K)
    x0 = g.uniform(-cfg.start_x_range, cfg.start_x_range, n_eps).astype(np.float32)
    pos = torch.as_tensor(np.stack([x0, np.full_like(x0, cfg.start_y)], 1), device=device)
    distract = torch.as_tensor(g.standard_normal((n_eps, cfg.distractor_dim)).astype(np.float32), device=device)
    goal = torch.as_tensor(GOAL, device=device)
    alive = torch.ones(n_eps, dtype=torch.bool, device=device)
    success = torch.zeros(n_eps, dtype=torch.bool, device=device)
    crashed = torch.zeros(n_eps, dtype=torch.bool, device=device)
    cross_x = torch.full((n_eps,), float("nan"), device=device)
    act_dim = cfg.horizon * 2
    torch.manual_seed(20_000 + seed)
    steps = 0
    while steps < MAX_STEPS and alive.any():
        obs = torch.cat([pos, distract], 1)
        a = an.u(sample(net, method, on.n(obs), act_dim, flow_steps, stochastic))
        a = a.view(n_eps, cfg.horizon, 2)
        for j in range(cfg.n_action_exec):
            new = pos + a[:, j] * alive[:, None]
            blocked = alive & segment_hits_wall(pos, new, centers)
            # x at the wall's centreline, to record which gap was used
            y0, y1 = pos[:, 1], new[:, 1]
            crosses = alive & (torch.sign(y0) != torch.sign(y1)) & (y0 != y1)
            frac = (-y0 / (y1 - y0 + 1e-12)).clamp(0, 1)
            xc = pos[:, 0] + frac * (new[:, 0] - pos[:, 0])
            first_cross = crosses & torch.isnan(cross_x)
            cross_x = torch.where(first_cross, xc, cross_x)
            crashed |= blocked
            pos = torch.where(blocked[:, None], pos, new)
            reached = alive & ~blocked & ((pos - goal).norm(dim=1) < GOAL_TOL)
            success |= reached
            alive &= ~(blocked | reached)
            steps += 1
    # which gap did successful episodes use
    gap_idx = torch.full((n_eps,), -1, device=device, dtype=torch.long)
    ok = ~torch.isnan(cross_x)
    c = torch.as_tensor(centers, device=device)
    gap_idx[ok] = torch.argmin((cross_x[ok, None] - c).abs(), 1)
    gap_idx[~success] = -1
    return {
        "success": success.float().mean().item(),
        "crash": crashed.float().mean().item(),
        "timeout": (~success & ~crashed).float().mean().item(),
        "gap_idx": gap_idx.cpu().numpy(),
        "x0": x0,
    }


N_X0_BINS = 10


def expert_conditional(x0: np.ndarray, cfg: TaskCfg) -> np.ndarray:
    """Expert P(gap | x0) = alpha * p + (1 - alpha) * onehot(nearest gap)."""
    centers = gap_centers(cfg.K)
    p = np.array(cfg.mode_probs or [1.0 / cfg.K] * cfg.K)
    near = np.argmin(np.abs(x0[:, None] - centers), 1)
    return cfg.alpha * p[None] + (1 - cfg.alpha) * np.eye(cfg.K)[near]


def mode_stats(res, cfg: TaskCfg):
    """Distributional fidelity of the policy's mode choice, *conditional on the
    start position*.

    Start positions are binned (bin edges are symmetric, so the nearest-gap
    boundary x0=0 is a bin edge). In each bin we compare the policy's gap
    distribution among successful episodes to the expert's conditional P(gap|x0),
    and report the success-weighted mean total-variation distance.

    A marginal (pooled) TV is misleading here: a regression policy that always
    takes the nearest gap still yields a 50/50 pooled split.

    Also returns `counts`: (bins, K+1) matrix of outcomes per x0 bin (last
    column = failure), so any metric can be recomputed later without retraining.
    """
    x0, gi = res["x0"], res["gap_idx"]
    edges = np.linspace(-cfg.start_x_range, cfg.start_x_range, N_X0_BINS + 1)
    b = np.clip(np.digitize(x0, edges) - 1, 0, N_X0_BINS - 1)
    counts = np.zeros((N_X0_BINS, cfg.K + 1), dtype=int)
    np.add.at(counts, (b, np.where(gi >= 0, gi, cfg.K)), 1)
    exp = expert_conditional(x0, cfg)
    tvs, ws = [], []
    for j in range(N_X0_BINS):
        m = (b == j) & (gi >= 0)
        if m.sum() == 0:
            continue
        pol = np.bincount(gi[m], minlength=cfg.K) / m.sum()
        tvs.append(0.5 * np.abs(pol - exp[b == j].mean(0)).sum())
        ws.append(m.sum())
    tv = float(np.average(tvs, weights=ws)) if ws else float("nan")
    succ = gi >= 0
    hist = np.bincount(gi[succ], minlength=cfg.K) / max(succ.sum(), 1)
    return {"tv": tv, "hist": hist, "counts": counts}
