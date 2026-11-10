"""Run the official training script over demos x method x seed.

Every launch is an immutable attempt directory, used as the working directory:
the official logger keeps checkpoints in ./checkpoints under a name without the
dataset size and appends to metrics.jsonl, so sharing a directory could load a
checkpoint from another subset or mix attempts. A cell is complete only when an
attempt holds a DONE.json, written after exit code 0 and validation. A DONE.json
made with other settings or data stops the launch instead of being reused.

Jobs are ordered seed by seed, so stopping early leaves complete replicates.

Usage: python run_grid.py --root ~/mip-density --tag sweep [--per_gpu 3]
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import itertools
import json
import math
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

METHODS = ("flow", "mip", "regression")
PRIMARY = {"flow": "mean_success_9", "mip": "mean_success_1", "regression": "mean_success_1"}


def dataset_for(root: Path, n: int, seed: int) -> Path:
    if n == 200:
        return root / "data/tool_hang_ph_low_dim_abs.hdf5"
    return root / f"data/subsets/tool_hang_N{n}_s{seed}.hdf5"


@functools.lru_cache(maxsize=None)
def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def overrides(root: Path, n: int, method: str, seed: int, attempt: Path, compile: bool = True) -> list[str]:
    """Hydra overrides shared by training, final evaluation and preflight."""
    return [
        "task=tool_hang_ph_state", "network=chiunet",
        f"optimization.loss_type={method}", f"optimization.seed={seed}",
        "optimization.auto_resume=false",
        # official option, about 13% faster; off is the official default
        *(["optimization.use_compile=true", "optimization.compile_mode=reduce-overhead"] if compile
          else ["optimization.use_compile=false"]),
        "~task.dataset_repo", "~task.dataset_filename", f"+task.dataset_path={dataset_for(root, n, seed)}",
        f"log.log_dir={attempt}/logs", f"log.exp_name=N{n}_{method}_s{seed}", "log.wandb_mode=disabled",
        "hydra.job.chdir=false", f"hydra.run.dir={attempt}/hydra",
    ]


def validate(attempt: Path, method: str, steps: int, eval_freq: int) -> str | None:
    """Returns a problem description, or None if the attempt is complete and clean."""
    f = attempt / "logs/metrics.jsonl"
    if not f.exists():
        return "no metrics.jsonl"
    key = PRIMARY[method]
    evals = [r for r in map(json.loads, f.open()) if key in r]
    expected = [eval_freq * (i + 1) - 1 for i in range(steps // eval_freq)]
    got = [r["step"] for r in evals]
    if got != expected:
        return f"evaluation steps {got} != expected {expected}"
    if any(not math.isfinite(r[key]) for r in evals):
        return "non-finite success"
    if not (attempt / "logs/models/model_latest.pt").exists():
        return "no model_latest.pt"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/mip-density")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--per_gpu", type=int, default=1)
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--eval_freq", type=int, default=60_000)
    ap.add_argument("--save_freq", type=int, default=60_000)
    ap.add_argument("--eval_episodes", type=int, default=40)
    ap.add_argument("--num_envs", type=int, default=5)  # 10 workers per run already reach the 64 GB account cap on 8 GPUs
    ap.add_argument("--sizes", default="200,25")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--max_failures", type=int, default=3)
    ap.add_argument("--no_compile", action="store_true")
    args = ap.parse_args()
    # model_latest.pt must be the last step, and the official eval counts episodes in whole batches of num_envs
    assert args.steps % args.save_freq == 0 and args.steps % args.eval_freq == 0
    assert args.eval_episodes % args.num_envs == 0
    root = Path(args.root).expanduser().resolve()
    py = root / "mado/.venv/bin/python"
    script = root / "mado/examples/train_robomimic.py"

    def manifest(n: int, method: str, seed: int) -> dict:
        data = dataset_for(root, n, seed)
        return {"n_demos": n, "method": method, "seed": seed, "steps": args.steps, "eval_freq": args.eval_freq,
                "save_freq": args.save_freq, "eval_episodes": args.eval_episodes, "num_envs": args.num_envs,
                "dataset": str(data), "compile": not args.no_compile,
                "dataset_md5": md5(data)}

    jobs: queue.Queue = queue.Queue()
    for seed, n, method in itertools.product(map(int, args.seeds.split(",")), map(int, args.sizes.split(",")),
                                             args.methods.split(",")):
        cell = root / f"runs/{args.tag}/N{n}_{method}_s{seed}"
        want = manifest(n, method, seed)
        done = sorted(cell.glob("attempt_*/DONE.json"))
        for d in done:
            have = json.loads(d.read_text())
            if {k: have.get(k) for k in want} != want:
                raise SystemExit(f"{d} was made with other settings: {have} vs {want}")
        if not done:
            jobs.put((n, method, seed, cell))
    print(jobs.qsize(), "runs to do", flush=True)

    failures: list[str] = []
    lock = threading.Lock()

    def run(gpu: str, n: int, method: str, seed: int, cell: Path) -> None:
        with lock:
            cell.mkdir(parents=True, exist_ok=True)
            k = len(list(cell.glob("attempt_*")))
            attempt = cell / f"attempt_{k}"
            attempt.mkdir()
        cmd = [str(py), str(script), *overrides(root, n, method, seed, attempt, not args.no_compile),
               f"optimization.device={args.device}", f"optimization.gradient_steps={args.steps}",
               f"task.num_envs={args.num_envs}", f"log.eval_freq={args.eval_freq}", f"log.save_freq={args.save_freq}",
               f"log.eval_episodes={args.eval_episodes}"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, WANDB_MODE="disabled", HF_HUB_OFFLINE="1", MUJOCO_GL="egl")
        t0 = time.time()
        with (attempt / "run.log").open("w") as log:
            code = subprocess.call(cmd, cwd=attempt, env=env, stdout=log, stderr=subprocess.STDOUT)
        problem = f"exit code {code}" if code else validate(attempt, method, args.steps, args.eval_freq)
        name = f"N={n} {method} s={seed} {attempt.name}"
        if problem:
            (attempt / "FAILED.txt").write_text(problem)
            with lock:
                failures.append(name)
            print(f"gpu{gpu} {name} FAILED: {problem}", flush=True)
            return
        ckpt = attempt / "logs/models/model_latest.pt"
        (attempt / "DONE.json").write_text(json.dumps({
            **manifest(n, method, seed), "ckpt_md5": md5(ckpt), "cmd": cmd, "hours": (time.time() - t0) / 3600}))
        # only model_latest.pt is evaluated; each of these copies is another 318 MB
        shutil.rmtree(attempt / "checkpoints", ignore_errors=True)
        (attempt / "logs/models/model_best.pt").unlink(missing_ok=True)
        print(f"gpu{gpu} {name} done in {(time.time() - t0) / 3600:.2f} h", flush=True)

    def worker(gpu: str) -> None:
        while len(failures) < args.max_failures:
            try:
                job = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                run(gpu, *job)
            except Exception as e:  # a launcher error counts as a failure, it must not kill the worker silently
                with lock:
                    failures.append(f"{job[:3]}: {e!r}")
                print("launcher error", job[:3], repr(e), flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus.split(",") for _ in range(args.per_gpu)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failures:
        raise SystemExit(f"{len(failures)} failures (stopped at {args.max_failures}): {failures}")
    if not jobs.empty():
        raise SystemExit("jobs left in the queue")
    print("all done", flush=True)


if __name__ == "__main__":
    main()
