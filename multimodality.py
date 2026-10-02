"""How multimodal are demonstration actions given the observation?

For a sample of query states we find, in every *other* demonstration, the most
similar state (so temporal neighbours from the same trajectory can never count),
keep the k closest demos, and test whether their action chunks form two
separated clusters.

Statistic (per query state), on the k neighbour chunks:
  * project on each of the top `n_pc` principal components of the chunks,
  * optimal 1-D 2-means split (min cluster size >= 20% of k),
  * Ashman's D = |mu1 - mu2| / sqrt((s1^2 + s2^2) / 2); take the max over PCs.
A state is flagged "multimodal" if D exceeds the 99th percentile of the same
statistic under a *uniform* (flat, unimodal) spread of k samples -- i.e. a wide
unimodal cloud does not count, only two clusters with a gap.

Every dataset is analysed under a grid of settings, because each of them
changes the answer (see README, "What the metric can and cannot say"):
  obs_steps  1 or 2 consecutive observations define "the same state"
  gripper    Robomimic only: include the binary gripper command or not
  n_pc       1 or 3 principal directions tested
  inject     positive control: every demo's actions are shifted by a random
             sign * `inject` action-stds, either along the first action
             dimension ("axis") or along a random direction in action space
             ("random"), so two modes 2*inject apart exist by construction

Reported:
  frac_mm   fraction of query states flagged multimodal (chance level = 1%)
  sep       mean separation of flagged states (action-std units, per-step RMS)
  locality  median distance to the k-th neighbour demo / median distance
            between random states (small = the neighbours really are similar)

Known blind spots: modes whose separation is small relative to the spread along
the top principal directions, minority modes below 20% of the neighbours, and
more than two modes.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

H_DEFAULT = 8  # action chunk length (executed horizon in the paper's configs)
MIN_FRAC = 0.2


# ----------------------------------------------------------------------------
# Loaders: each returns a list of (obs[T, do], act[T, da]) per demo
# ----------------------------------------------------------------------------
def rotvec_to_6d(rv: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    return R.from_rotvec(rv).as_matrix()[:, :, :2].reshape(len(rv), 6)


def quat_to_6d(q: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    return R.from_quat(q).as_matrix()[:, :, :2].reshape(len(q), 6)  # robosuite quats are (x, y, z, w)


def load_robomimic(path: str):
    """Actions are [pos(3), rot6d(6), gripper(1)]; the gripper is the last dim."""
    import h5py

    demos = []
    with h5py.File(path, "r") as f:
        for k in sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1])):
            d = f["data"][k]
            o = np.concatenate(
                [
                    d["obs"]["object"][:],
                    d["obs"]["robot0_eef_pos"][:],
                    quat_to_6d(d["obs"]["robot0_eef_quat"][:]),
                    d["obs"]["robot0_gripper_qpos"][:],
                ],
                1,
            )
            a = d["actions"][:]
            a = np.concatenate([a[:, :3], rotvec_to_6d(a[:, 3:6]), a[:, 6:7]], 1)
            demos.append((o.astype(np.float32), a.astype(np.float32)))
    return demos


def load_pusht(path: str):
    import zarr

    store = zarr.ZipStore(path, mode="r")
    # the archive nests everything under "<name>.zarr/"
    g = zarr.open(store, mode="r", path=Path(path).name.removesuffix(".zip"))
    state = g["data"]["state"][:]
    act = g["data"]["action"][:]
    ends = g["meta"]["episode_ends"][:]
    obs = np.concatenate(
        [state[:, :4] / 512.0, np.sin(state[:, 4:5]), np.cos(state[:, 4:5])], 1
    ).astype(np.float32)
    act = (act / 512.0).astype(np.float32)
    demos, s = [], 0
    for e in ends:
        demos.append((obs[s:e], act[s:e]))
        s = e
    return demos


def load_toy(alpha: float, n_demos: int = 250, seed: int = 0, **kw):
    from mm_nav import GOAL, STEP, TaskCfg, _polyline_rollout, gap_centers, sample_mode

    cfg = TaskCfg(alpha=alpha, n_demos=n_demos, **kw)
    rng = np.random.default_rng(seed)
    centers = gap_centers(cfg.K)
    demos = []
    for _ in range(cfg.n_demos):
        x0 = rng.uniform(-cfg.start_x_range, cfg.start_x_range)
        k = sample_mode(rng, x0, cfg, centers)
        cx = centers[k] + rng.uniform(-cfg.jitter, cfg.jitter)
        pts = np.array(
            [[x0, cfg.start_y], [cx, -cfg.approach_y + rng.uniform(-0.05, 0.05)],
             [cx, cfg.approach_y + rng.uniform(-0.05, 0.05)], GOAL], dtype=np.float32)
        traj = _polyline_rollout(pts, STEP)
        demos.append((traj[:-1], np.diff(traj, axis=0)))
    return demos


# ----------------------------------------------------------------------------
# Statistic
# ----------------------------------------------------------------------------
def ashman_split(p: np.ndarray, min_frac: float = MIN_FRAC):
    """Best 1-D 2-means split of p (k,). Returns (D, separation)."""
    p = np.sort(p)
    k = len(p)
    m = max(2, int(np.ceil(min_frac * k)))
    c1 = np.cumsum(p)
    c2 = np.cumsum(p * p)
    best = None
    for i in range(m, k - m + 1):  # left = p[:i], right = p[i:]
        n1, n2 = i, k - i
        s1, s2 = c1[i - 1], c1[-1] - c1[i - 1]
        q1, q2 = c2[i - 1], c2[-1] - c2[i - 1]
        w = (q1 - s1 * s1 / n1) + (q2 - s2 * s2 / n2)
        if best is None or w < best[0]:
            best = (w, i)
    i = best[1]
    a, b = p[:i], p[i:]
    sep = b.mean() - a.mean()
    pooled = np.sqrt((a.var() + b.var()) / 2) + 1e-12
    return sep / pooled, sep


_NULL_CACHE: dict[tuple[int, int], float] = {}


def null_threshold(k: int, n_pc: int, q: float = 0.99, n: int = 4000) -> float:
    """q-quantile of max-over-n_pc Ashman's D for k samples from a uniform
    (flat, unimodal) spread in each direction."""
    if (k, n_pc) not in _NULL_CACHE:
        rng = np.random.default_rng(0)
        _NULL_CACHE[(k, n_pc)] = float(np.quantile(
            [max(ashman_split(rng.uniform(size=k))[0] for _ in range(n_pc)) for _ in range(n)], q))
    return _NULL_CACHE[(k, n_pc)]


def build(demos, H: int, obs_steps: int):
    """Stack per-timestep (obs history, action chunk [H, da], demo id, phase)."""
    O, A, D, P = [], [], [], []
    for di, (o, a) in enumerate(demos):
        T = len(a) - H + 1
        if T <= obs_steps:
            continue
        idx = np.arange(obs_steps - 1, T)
        O.append(np.concatenate([o[idx - j] for j in range(obs_steps)], 1))
        A.append(np.stack([a[i : i + H] for i in idx]))
        D.append(np.full(len(idx), di))
        P.append(idx / max(len(a) - 1, 1))
    return np.concatenate(O), np.concatenate(A), np.concatenate(D), np.concatenate(P)


def neighbours(demos, H: int = H_DEFAULT, obs_steps: int = 2, k: int = 16, n_query: int = 3000,
               seed: int = 0, device: str = "cpu"):
    """Cross-demo nearest neighbours in observation space (computed once per
    dataset and observation history; the statistic variants reuse it)."""
    O, A, D, P = build(demos, H, obs_steps)
    O = (O - O.mean(0)) / (O.std(0) + 1e-6)
    n_demo = int(D.max()) + 1
    k = min(k, n_demo - 1)
    rng = np.random.default_rng(seed)
    q_idx = rng.choice(len(O), min(n_query, len(O)), replace=False)
    Ot = torch.as_tensor(O, device=device)
    starts = np.flatnonzero(np.r_[True, D[1:] != D[:-1]])  # contiguous demo segments
    ends = np.r_[starts[1:], len(D)]
    seg_demo = D[starts]
    r = rng.choice(len(O), (2000, 2))
    rand_d = float(np.median(np.linalg.norm(O[r[:, 0]] - O[r[:, 1]], axis=1)))
    nbs, kth = [], []
    for b in range(0, len(q_idx), 256):
        qi = q_idx[b : b + 256]
        dist = torch.cdist(Ot[qi], Ot)
        mins = torch.empty(len(qi), len(starts), device=device)
        args = torch.empty(len(qi), len(starts), dtype=torch.long, device=device)
        for j, (s, e) in enumerate(zip(starts, ends)):
            m, am = dist[:, s:e].min(1)
            mins[:, j], args[:, j] = m, am + s
        mins[torch.as_tensor(seg_demo[None, :] == D[qi][:, None], device=device)] = float("inf")
        top_d, top_j = mins.topk(k, dim=1, largest=False)
        nbs.append(args.gather(1, top_j).cpu().numpy())
        kth.append(top_d[:, -1].cpu().numpy())
    return {"A": A, "D": D, "nb": np.concatenate(nbs), "phase": P[q_idx], "k": k, "n_demo": n_demo,
            "n_states": int(len(O)), "obs_dim": int(O.shape[1]),
            "locality": float(np.median(np.concatenate(kth)) / rand_d)}


def statistic(nb, dims=None, n_pc: int = 1, inject: float = 0.0, inject_dir: str = "axis", seed: int = 0):
    """Bimodality of the neighbours' action chunks, restricted to action `dims`."""
    A, D = nb["A"], nb["D"]
    if dims is not None:
        A = A[:, :, dims]
    H, da = A.shape[1], A.shape[2]
    A = A / (A.std((0, 1)) + 1e-6)  # per action dimension
    if inject > 0:
        rng = np.random.default_rng(seed + 1)
        sign = rng.choice([-1.0, 1.0], nb["n_demo"])
        u = np.zeros(da)
        u[0] = 1.0
        if inject_dir == "random":
            u = rng.standard_normal(da)
            u /= np.linalg.norm(u)
        A = A + (sign[D] * inject)[:, None, None] * u[None, None, :]
    A = A.reshape(len(A), -1)
    thr = null_threshold(nb["k"], n_pc)
    Ds, seps = [], []
    for row in nb["nb"]:
        X = A[row]
        X = X - X.mean(0)
        pcs = np.linalg.svd(X, full_matrices=False)[2][:n_pc]
        d, sep = max(ashman_split(X @ pc) for pc in pcs)
        Ds.append(d)
        seps.append(sep / np.sqrt(H))
    Ds, seps, ph = np.array(Ds), np.array(seps), nb["phase"]
    flag = Ds > thr
    return {
        "frac_mm": float(flag.mean()),
        "sep": float(seps[flag].mean()) if flag.any() else 0.0,
        "median_D": float(np.median(Ds)),
        "threshold": thr,
        "frac_mm_by_phase": [float(flag[(ph >= lo) & (ph < lo + 0.2)].mean()) for lo in (0.0, 0.2, 0.4, 0.6, 0.8)],
    }


# ----------------------------------------------------------------------------
ROBOMIMIC = {
    **{f"{t}_{v}": f"data/robomimic/{t}/{v}/low_dim_abs.hdf5"
       for t, v in [("lift", "ph"), ("lift", "mh"), ("can", "ph"), ("can", "mh"),
                    ("square", "ph"), ("square", "mh"), ("tool_hang", "ph")]},
    # Tool-Hang datasets recollected by the paper's authors for their dataset-quality
    # ablation (their Table 11): rollouts of trained policies with injected delay / noise.
    **{f"tool_hang_{v}": f"data/robomimic/tool_hang/{v}/low_dim_abs.hdf5"
       for v in ("good_policy", "delay_policy", "noisydelay_policy")},
}
CONTROLS = [(0.0, "none"), (1.0, "axis"), (1.0, "random")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="toy,pusht,robomimic")
    ap.add_argument("--only", default=None, help="comma-separated dataset names")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--H", type=int, default=H_DEFAULT)
    ap.add_argument("--out", default="results/multimodality.json")
    args = ap.parse_args()
    out = Path(args.out)
    rows = []

    def run(name, demos, dim_sets):
        for obs_steps in (1, 2):
            nb = neighbours(demos, H=args.H, obs_steps=obs_steps, k=args.k)
            for (grip, dims), n_pc, (inj, inj_dir) in itertools.product(dim_sets, (1, 3), CONTROLS):
                r = statistic(nb, dims=dims, n_pc=n_pc, inject=inj, inject_dir=inj_dir)
                rows.append({"dataset": name, "obs_steps": obs_steps, "gripper": grip, "n_pc": n_pc,
                             "inject": inj, "inject_dir": inj_dir, "n_demos": nb["n_demo"],
                             "n_states": nb["n_states"], "k": nb["k"], "locality": nb["locality"], **r})
                print(f"{name:28s} obs={obs_steps} grip={str(grip):5s} pc={n_pc} inj={inj_dir:6s} "
                      f"flagged={100 * r['frac_mm']:5.1f}%  locality={nb['locality']:.2f}", flush=True)
            out.write_text(json.dumps(rows, indent=1))

    only = set(args.only.split(",")) if args.only else None
    if "toy" in args.which:
        for a in (0.0, 0.25, 0.5, 0.75, 1.0):
            run(f"toy_alpha{a}", load_toy(a), [(None, None)])
    if "pusht" in args.which and (only is None or "pusht" in only):
        run("pusht", load_pusht("data/pusht/pusht_cchi_v7_replay.zarr.zip"), [(None, None)])
    if "robomimic" in args.which:
        for name, path in ROBOMIMIC.items():
            if only is None or name in only:
                run(name, load_robomimic(path), [(False, list(range(9))), (True, None)])


if __name__ == "__main__":
    main()
