"""CPU-only checks before any GPU use. Exits non-zero on the first failure.

Checks, using the same Hydra overrides as run_grid.py:
  1. overrides compose and remove the Hugging Face fields, so that both dataset
     and environment use the local file;
  2. subsets: demo count, nesting, data identical to the source;
  3. official dataset on the smallest and the full file: dimensions, windows >=
     batch size, finite normalisation, share of full-data observations outside
     the subset's normalisation range;
  4. environment: observation dimension matches the data; a seeded reset gives
     the same state when repeated, and different seeds give different states;
  5. one training update and one sample on CPU for each method.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
ROOT = Path(os.environ.get("MIP_ROOT", "~/mip-density")).expanduser().resolve()
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT / "mado/examples")]

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from run_grid import METHODS, dataset_for, overrides  # noqa: E402


def cfg_for(n, method, seed, tmp, num_envs=2):
    ov = [o for o in overrides(ROOT, n, method, seed, tmp) if not o.startswith("hydra.")]
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "mado/examples/configs")):
        c = compose(config_name="main", overrides=ov + ["optimization.device=cpu", f"task.num_envs={num_envs}"])
    return c


def main() -> None:
    tmp = ROOT / "runs/preflight"
    tmp.mkdir(parents=True, exist_ok=True)
    os.chdir(tmp)

    # 1. config
    c = cfg_for(25, "flow", 0, tmp)
    assert not hasattr(c.task, "dataset_repo") and not hasattr(c.task, "dataset_filename"), "HF fields still present"
    assert Path(c.task.dataset_path).is_absolute() and Path(c.task.dataset_path).exists()
    print("config ok:", c.task.dataset_path, "| act_steps", c.task.act_steps, "| obs_steps", c.task.obs_steps,
          "| sample_mode", c.optimization.sample_mode, "| ema", c.optimization.ema_rate)

    # 2. subsets
    src = dataset_for(ROOT, 200, 0)
    with h5py.File(src, "r") as f:
        assert sorted(f["data"], key=lambda s: int(s.split("_")[1])) == [f"demo_{i}" for i in range(200)]
        lengths = {i: int(f["data"][f"demo_{i}"].attrs["num_samples"]) for i in range(200)}
    for seed in (0, 1, 2):
        prev: set = set()
        for n in (25, 50, 100):
            path = dataset_for(ROOT, n, seed)
            ids = json.loads(path.with_suffix(".json").read_text())["demo_ids"]
            assert len(ids) == len(set(ids)) == n and prev <= set(ids), "not nested"
            prev = set(ids)
            with h5py.File(path, "r") as f:
                assert len(f["data"]) == n
                assert int(f["data"].attrs["total"]) == sum(lengths[i] for i in ids)
    print("subsets ok: nested, counts and totals match")

    # 3. official dataset
    import train_robomimic as T  # official module

    data = {}
    for n in (25, 200):
        c = cfg_for(n, "flow", 0, tmp)
        c.task.horizon = 16  # the official main() rounds up for Chi-UNet
        d = T.make_dataset(c.task)
        data[n] = d
        norm = d.normalizer["obs"]["state"]
        print(f"N={n}: windows {len(d)}, steps {d.replay_buffer.n_steps}, episodes {d.replay_buffer.n_episodes}, "
              f"obs {d.replay_buffer['obs'].shape[-1]}, action {d.replay_buffer['action'].shape[-1]}")
        assert len(d) >= c.optimization.batch_size, "fewer windows than one batch"
        assert np.isfinite(norm.min).all() and np.isfinite(norm.range).all()
        s = d[0]
        assert s["action"].shape == (16, 10) and torch.isfinite(s["action"]).all()
    full = data[200].replay_buffer["obs"][:]
    for n, seed in itertools.product((25, 50), (0, 1, 2)):  # every subset the sweep uses
        c = cfg_for(n, "flow", seed, tmp)
        c.task.horizon = 16
        d = T.make_dataset(c.task)
        assert len(d) >= c.optimization.batch_size, "fewer windows than one batch"
        assert len(d) == d.replay_buffer.n_steps - 7 * d.replay_buffer.n_episodes, "window count"
        norm = d.normalizer["obs"]["state"]
        out = (full < norm.min) | (full > norm.max)
        print(f"N={n} s={seed}: windows {len(d)}; full-data steps with a coordinate outside this subset's min-max "
              f"range {out.any(1).mean():.1%}, values {out.mean():.1%}")

    # 4. environment
    c = cfg_for(25, "flow", 0, tmp)
    c.task.horizon = 16
    envs = T.make_vec_env(c.task, seed=0)
    o0, _ = envs.reset()
    assert o0.shape[-1] == data[25].replay_buffer["obs"].shape[-1], "env and data observation sizes differ"
    a, _ = envs.reset(seed=[5000, 5001])
    envs.step(data[25].undo_transform_action(data[25].replay_buffer["action"][:8][None].repeat(2, 0)))
    b, _ = envs.reset(seed=[5000, 5001])
    d, _ = envs.reset(seed=[6000, 6001])
    assert np.allclose(a, b, atol=1e-6), "same seed, different initial state"
    assert not np.allclose(a, d, atol=1e-6), "different seed, same initial state"
    envs.close()
    print("environment ok: obs", o0.shape, "seeded resets repeat, different seeds differ")

    # 4b. the real bank with 20 workers, and episode accounting as in the official eval
    c = cfg_for(25, "flow", 0, tmp, num_envs=20)
    c.task.horizon = 16
    envs = T.make_vec_env(c.task, seed=0)
    envs.reset()
    n = envs.num_envs

    def bank_hashes(batch):
        seeds = [100_000 + batch * n + i for i in range(n)]
        obs, _ = envs.reset(seed=seeds)
        return [hashlib.md5(np.round(np.asarray(o, dtype=np.float64), 6).tobytes()).hexdigest() for o in obs]

    hashes = [h for batch in range(20) for h in bank_hashes(batch)]
    assert len(set(hashes)) == 400, f"only {len(set(hashes))} distinct initial states out of 400"
    assert bank_hashes(0) == hashes[:n] and bank_hashes(7) == hashes[7 * n:8 * n], "the bank does not repeat"
    print("bank ok: 400 distinct initial observations, repeatable with 20 workers")

    bank_hashes(0)
    act = data[25].undo_transform_action(data[25].replay_buffer["action"][:8][None].repeat(n, 0))
    t = 0
    while t < c.task.max_episode_steps:  # the official loop
        _, _, term, trunc, _ = envs.step(act)
        t += c.task.act_steps
        if t < c.task.max_episode_steps:
            assert not (term | trunc).any(), f"episode ended early at {t}"
    assert (term | trunc).all(), "episode did not end at the horizon"
    assert bank_hashes(0) == hashes[:n], "seeded reset after a finished episode differs"
    envs.close()
    print(f"episode accounting ok: every episode runs to the horizon ({c.task.max_episode_steps}), no autoreset inside")

    # 5. one update and one sample per method, on CPU
    for method in METHODS:
        c = cfg_for(25, method, 0, tmp)
        OmegaConf.set_struct(c, False)
        c.task.horizon = 16
        c.task.obs_dim = int(o0.shape[-1])
        agent = T.TrainingAgent(c)
        loader = torch.utils.data.DataLoader(data[25], batch_size=8, shuffle=True)
        batch = next(iter(loader))
        obs = batch["obs"]["state"][:, : c.task.obs_steps]
        info = agent.update(batch["action"][:, :16], obs, torch.zeros(8))
        act = agent.sample(act_0=torch.randn(8, 16, 10), obs={"state": obs}, num_steps=1, use_ema=True)
        assert act.shape == (8, 16, 10) and torch.isfinite(act).all()
        print(f"{method}: update ok, loss {float(info.get('loss', float('nan'))):.3f}, "
              f"params {sum(p.numel() for p in agent.flow_map.parameters()) / 1e6:.1f} M")
    print("PREFLIGHT PASSED")


if __name__ == "__main__":
    main()
