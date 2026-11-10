#!/usr/bin/env bash
# One command per stage, each with its own tag so that short runs cannot stand in for full ones.
# Usage: bash stage.sh cputest|smoke|pilot|sweep [per_gpu] [sizes]|evals [per_gpu] [sizes]|nocompile|analyze [sizes]
set -eu
ROOT="$HOME/mip-density"
PY="$ROOT/mado/.venv/bin/python"
D="$(cd "$(dirname "$0")" && pwd)"
# the home quota can be full, so nothing here may need to write to home
LOCAL=/var/tmp/bsuvernev2-mip-runs
export PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME="$LOCAL/.cache" MPLCONFIGDIR="$LOCAL/.mpl" TRITON_CACHE_DIR="$LOCAL/.triton"
# torch.compile otherwise keeps a pool of compile workers per job, which costs memory under the account cap
export TORCHINDUCTOR_COMPILE_THREADS=1

# The account is capped at 64 GB and hitting the cap kills unrelated processes, so stop our own jobs first.
guard() {
  local C="/sys/fs/cgroup/user.slice/user-$(id -u).slice"
  while sleep 20; do
    if [ "$(awk '/^anon /{print int($2 / 2^30)}' "$C/memory.stat")" -ge 56 ]; then
      echo "$(date) memory guard: account above 56 GB, stopping our jobs" >> "$LOCAL/logs/guard.log"
      pkill -u "$(id -u)" -f "mip-density/mado/examples/train_robomimic|density_real/eval_final" || true
    fi
  done
}
guard &
GUARD=$!
GPUS=0,1,2,3,4,5,6,7
trap 'kill $GUARD 2>/dev/null || true' EXIT

case "$1" in
  cputest)  # the whole pipeline without any GPU
    $PY "$D/run_grid.py" --root "$ROOT" --tag cputest --gpus "" --device cpu --per_gpu 3 --steps 4 --eval_freq 4 --save_freq 4 --eval_episodes 20 --sizes 25 --seeds 0
    $PY "$D/run_evals.py" --root "$ROOT" --tag cputest --gpus "" --device cpu --per_gpu 3 --episodes 20 --stochastic_sizes 25
    $PY "$D/analyze.py" --root "$ROOT" --tag cputest --sizes 25 --seeds 0 --steps 4 --eval_freq 4 --eval_episodes 20 --episodes 20
    ;;
  smoke)
    TAG="${2:-smoke}"
    $PY "$D/run_grid.py" --root "$ROOT" --tag "$TAG" --steps 200 --eval_freq 200 --save_freq 200 --eval_episodes 20 --sizes 25 --seeds 0
    $PY "$D/run_evals.py" --root "$ROOT" --tag "$TAG" --episodes 20
    $PY "$D/analyze.py" --root "$ROOT" --tag "$TAG" --sizes 25 --seeds 0 --steps 200 --eval_freq 200 --eval_episodes 20 --episodes 20
    ;;
  pilot)
    $PY "$D/pilot.py" --root "$ROOT"
    ;;
  load)  # 8 jobs at once with evaluations, to measure memory before the sweep
    $PY "$D/run_grid.py" --root "$ROOT" --tag load --gpus "$GPUS" --steps 400 --eval_freq 400 --save_freq 400
    ;;
  sweep)
    $PY "$D/run_grid.py" --root "$ROOT" --tag sweep --gpus "$GPUS" --per_gpu "${2:-1}" --sizes "${3:-200,25}"
    ;;
  nocompile)  # control: MIP and regression at N=200 trained and evaluated without torch.compile
    $PY "$D/run_grid.py" --root "$ROOT" --tag nocompile --gpus "${TRAIN_GPUS:-0,1,2,4,5,6}" --no_compile --sizes 200 --methods mip,regression
    $PY "$D/run_evals.py" --root "$ROOT" --tag nocompile --gpus "${EVAL_GPUS:-0,1,2,4}" --episodes 400
    ;;
  evals)
    $PY "$D/run_evals.py" --root "$ROOT" --tag sweep --gpus "${EVAL_GPUS:-0,1,2}" --per_gpu "${2:-1}" --episodes 400 --stochastic_sizes "${3:-}"
    ;;
  analyze)
    $PY "$D/analyze.py" --root "$ROOT" --tag sweep --sizes "${2:-200,25}" --out "$LOCAL/results_sweep.jsonl"
    ;;
  *) echo "unknown stage $1"; exit 2 ;;
esac
