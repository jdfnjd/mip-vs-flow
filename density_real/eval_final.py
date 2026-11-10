"""Evaluate a final checkpoint on a fixed bank of initial states, unmodified official eval().

The official training-time evaluation does not pair methods: after the first
reset, the environment workers draw from their own random stream. Here the
official eval() receives a thin proxy whose reset() seeds every environment
explicitly, so every checkpoint starts from the same states. The proxy also keeps
each episode's seed, initial-observation hash and success, and these must agree
with the official mean. An episode ends only at step 700 (robomimic's is_done is
always False), so one reset is one episode. The model keeps the normaliser of its
own training subset because the dataset is rebuilt from the same file, whose hash
and the checkpoint's are checked against the run's DONE.json.

Usage: python eval_final.py <hydra overrides> +done=<DONE.json> +out=<result.json> +bank=100000
Hydra overrides are the same as in training; set log.eval_episodes (multiple of num_envs).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import torch

MADO = Path(os.environ["MADO_ROOT"]).resolve()
sys.path.insert(0, str(MADO / "examples"))
import train_robomimic as T  # noqa: E402

from mip.samplers import get_default_step_list  # noqa: E402


def file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


class SeededResets:
    """Passes everything to the vector env; reset() seeds a fixed bank, step() records rewards."""

    def __init__(self, envs, bank: int):
        self._envs, self._bank = envs, bank
        self.seeds: list[int] = []
        self.init_hashes: list[str] = []
        self.rewards: list[np.ndarray] = []

    def __getattr__(self, name):
        return getattr(self._envs, name)

    def reset(self, **kwargs):
        n = self._envs.num_envs
        seeds = [self._bank + len(self.seeds) + i for i in range(n)]
        obs, info = self._envs.reset(seed=seeds)
        self.seeds += seeds
        self.init_hashes += [hashlib.md5(np.round(np.asarray(o, dtype=np.float64), 6).tobytes()).hexdigest()
                             for o in obs]
        self.rewards.append(np.zeros(n))
        return obs, info

    def step(self, action):
        out = self._envs.step(action)
        self.rewards[-1] += out[1]
        return out


@hydra.main(version_base=None, config_path=str(MADO / "examples/configs"), config_name="main")
def main(config):
    done = Path(config.done)
    meta = json.loads(done.read_text())
    ckpt = done.parent / "logs/models/model_latest.pt"
    assert file_md5(ckpt) == meta["ckpt_md5"], "checkpoint differs from the completed run"
    assert file_md5(Path(config.task.dataset_path)) == meta["dataset_md5"], "dataset differs from the completed run"

    torch.set_float32_matmul_precision("high")
    T.set_seed(config.optimization.seed)
    T.limit_threads(1)
    logger = T.Logger(config)
    if config.network.network_type == "chiunet":  # as in the official main()
        config.task.horizon = int(2 ** np.ceil(np.log2(config.task.horizon)))
    envs = T.make_vec_env(config.task, seed=config.optimization.seed)
    obs, _ = envs.reset()
    config.task.obs_dim = obs.shape[-1]
    dataset = T.make_dataset(config.task)
    agent = T.TrainingAgent(config)
    agent.load(str(ckpt), load_optimizer=False)
    agent.eval()

    proxy = SeededResets(envs, int(config.bank))
    num_steps = int(get_default_step_list(config.optimization.loss_type)[0])  # flow: 9, regression and MIP: 1
    metrics = T.eval(config, proxy, dataset, agent, logger, num_steps)
    per_episode = [float(r > 0) for r in np.concatenate(proxy.rewards)]
    official = float(metrics[f"mean_success_{num_steps}"])
    assert len(per_episode) == int(config.log.eval_episodes), "wrong episode count"
    assert abs(float(np.mean(per_episode)) - official) < 1e-9, "per-episode results differ from the official mean"
    result = {"ckpt": str(ckpt), "ckpt_md5": meta["ckpt_md5"], "dataset": str(config.task.dataset_path),
              "dataset_md5": meta["dataset_md5"], "bank": int(config.bank), "num_steps": num_steps,
              "sample_mode": config.optimization.sample_mode, "episodes": len(per_episode), "success": official,
              "bank_seeds": proxy.seeds, "init_hashes": proxy.init_hashes, "episode_success": per_episode}
    Path(config.out).write_text(json.dumps(result))
    print("RESULT", official, len(per_episode), flush=True)
    envs.close()


if __name__ == "__main__":
    main()
