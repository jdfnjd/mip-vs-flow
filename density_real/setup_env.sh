#!/usr/bin/env bash
# Install the official environment from its lock file. Large git clones and tarballs
# are flaky here, so robosuite is rebuilt from the PyPI wheel and checked file by file
# against the commit pinned in uv.lock.
set -eu
ROOT="$HOME/mip-density"
# the home quota is 40 GB, so runs live on the local disk
mkdir -p /var/tmp/bsuvernev2-mip-runs
[ -e "$ROOT/runs" ] || ln -s /var/tmp/bsuvernev2-mip-runs "$ROOT/runs"
cd "$ROOT/mado"
UV=../uvenv/bin/uv
PY=.venv/bin/python
export UV_HTTP_TIMEOUT=900 UV_CONCURRENT_DOWNLOADS=4 CUDA_VISIBLE_DEVICES=
SHA=a071383d53568ab798eb315c0e95357911be922d
grep -q "robosuite.git?rev=v1.5.1#$SHA" uv.lock
[ -x $PY ] || $UV sync --python 3.12 --frozen --no-install-package robosuite
# the only change to the official code: it hardcodes OSMesa, which this server lacks (README: egl)
sed -i 's/^os.environ\["MUJOCO_GL"\] = "osmesa"/os.environ.setdefault("MUJOCO_GL", "osmesa")/' examples/train_robomimic.py
grep -q 'setdefault("MUJOCO_GL", "osmesa")' examples/train_robomimic.py

# each new curl resumes; curl's own --retry would discard progress
fetch() {  # url file size
  until [ "$(stat -c%s "$2" 2>/dev/null || echo 0)" = "$3" ]; do
    curl -sSL --connect-timeout 30 --speed-time 60 --speed-limit 1000 -C - -o "$2" "$1" || sleep 15
  done
}

WHL=/tmp/robosuite-1.5.1-py3-none-any.whl
fetch "https://files.pythonhosted.org/packages/f4/15/82093cadf23811463d0b52ec6745949356b66badf6e25bee64ec82aa8689/robosuite-1.5.1-py3-none-any.whl" $WHL 152011410
rm -rf /tmp/robosuite_src
$PY "$ROOT/density_real/rebuild_robosuite.py" $WHL /tmp/robosuite_src
$UV pip install --python $PY --no-deps --reinstall /tmp/robosuite_src
SITE=$($PY -c "import site; print(site.getsitepackages()[0])")
cp -r /tmp/robosuite_src/robosuite "$SITE/"
$PY "$ROOT/density_real/rebuild_robosuite.py" --verify "$SITE" | tee /tmp/rs_verify.txt
grep -q "not matching: \[\]" /tmp/rs_verify.txt
echo ENV_DONE

mkdir -p "$ROOT/data/subsets"
cd "$ROOT/data"
URL="https://huggingface.co/datasets/ChaoyiPan/mip-dataset/resolve/main/robomimic/tool_hang/ph/low_dim_abs.hdf5"
fetch "$URL" tool_hang_ph_low_dim_abs.hdf5 198601752
ETAG=$(curl -sI -m 60 "$URL" | tr -d '\r"' | awk 'tolower($1)=="x-linked-etag:"{print $2}')
echo "$ETAG  tool_hang_ph_low_dim_abs.hdf5" | sha256sum -c -
"$ROOT/mado/.venv/bin/python" "$ROOT/density_real/make_subsets.py" tool_hang_ph_low_dim_abs.hdf5 subsets tool_hang
echo ALL_DONE
