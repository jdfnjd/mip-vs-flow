"""Final evaluation of every completed run on one shared bank of initial states.

Primary: the official deterministic sampling (flow: z = 0, 9 steps), all done first.
Secondary and optional: flow with stochastic sampling at --stochastic_sizes.
Each job has its own working directory. A saved result that belongs to another
checkpoint, dataset or bank stops the launch instead of being reused.

Usage: python run_evals.py --root ~/mip-density --tag sweep --episodes 400 [--per_gpu 1]
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import threading
from pathlib import Path

from run_grid import overrides

BANK = 100_000  # far from the training seeds' environment seeds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/mip-density")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--per_gpu", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--stochastic_sizes", default="")
    args = ap.parse_args()
    assert args.episodes % 20 == 0
    root = Path(args.root).expanduser().resolve()
    py, script = root / "mado/.venv/bin/python", Path(__file__).resolve().parent / "eval_final.py"
    stochastic = {int(s) for s in args.stochastic_sizes.split(",") if s}

    primary, secondary = [], []
    for done in sorted((root / f"runs/{args.tag}").glob("*/attempt_*/DONE.json")):
        m = json.loads(done.read_text())
        modes = ["zero"]
        if m["method"] == "flow" and m["n_demos"] in stochastic:
            modes.append("stochastic")
        for mode in modes:
            out = done.parent / f"final_{mode}_{args.episodes}.json"
            if out.exists():
                r = json.loads(out.read_text())
                have = (r["ckpt_md5"], r["dataset_md5"], r["bank"], r["episodes"], r["sample_mode"])
                if have != (m["ckpt_md5"], m["dataset_md5"], BANK, args.episodes, mode):
                    raise SystemExit(f"{out} belongs to another checkpoint, dataset or bank")
            else:
                (primary if mode == "zero" else secondary).append((done, m, mode, out))
    jobs: queue.Queue = queue.Queue()
    for job in primary + secondary:
        jobs.put(job)
    print(len(primary), "primary and", len(secondary), "secondary evaluations to do", flush=True)
    failures: list[str] = []

    def worker(gpu: str) -> None:
        while True:
            try:
                done, m, mode, out = jobs.get_nowait()
            except queue.Empty:
                return
            d = done.parent / f"eval_{mode}_{args.episodes}"
            d.mkdir(exist_ok=True)
            # the same seed for every checkpoint; the states come from the bank anyway
            ov = [o for o in overrides(root, m["n_demos"], m["method"], m["seed"], d, m.get("compile", True))
                  if not o.startswith("optimization.seed=")]
            cmd = [str(py), str(script), *ov, "optimization.seed=0", f"optimization.sample_mode={mode}",
                   f"optimization.device={args.device}", f"log.eval_episodes={args.episodes}",
                   f"+done={done}", f"+out={out}", f"+bank={BANK}"]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, WANDB_MODE="disabled", HF_HUB_OFFLINE="1",
                       MUJOCO_GL="egl", MADO_ROOT=str(root / "mado"))
            with (d / "run.log").open("a") as log:
                code = subprocess.call(cmd, cwd=d, env=env, stdout=log, stderr=subprocess.STDOUT)
            if code or not out.exists():
                failures.append(str(out))
                print("FAILED", out, code, flush=True)
            else:
                print(json.loads(out.read_text())["success"], out, flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus.split(",") for _ in range(args.per_gpu)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failures:
        raise SystemExit(f"{len(failures)} evaluations failed")


if __name__ == "__main__":
    main()
