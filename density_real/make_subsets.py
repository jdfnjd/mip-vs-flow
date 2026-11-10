"""Nested demo subsets of a Robomimic low-dim dataset.

For each seed, a fixed random permutation of the demos is drawn and the first
n demos form the subset, so subsets are nested (25 in 50 in 100). The official
loader reads demo_0 .. demo_{k-1}, so demos are renumbered; the original ids
are recorded in <file>.json. Data attributes (env_args, used to build the
environment) are copied unchanged.

Usage: python make_subsets.py <source.hdf5> <out_dir> <name>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

SIZES = (25, 50, 100)
SEEDS = (0, 1, 2)


def write_subset(src: Path, dst: Path, ids: list[int]) -> None:
    with h5py.File(src, "r") as fi, h5py.File(dst, "w") as fo:
        d_in = fi["data"]
        d_out = fo.create_group("data")
        for k, v in d_in.attrs.items():
            d_out.attrs[k] = v
        total = 0
        for new, old in enumerate(ids):
            fi.copy(d_in[f"demo_{old}"], d_out, name=f"demo_{new}")
            total += int(d_in[f"demo_{old}"].attrs["num_samples"])
        d_out.attrs["total"] = total


def verify(src: Path, dst: Path, ids: list[int]) -> None:
    with h5py.File(src, "r") as fi, h5py.File(dst, "r") as fo:
        assert len(fo["data"]) == len(ids)
        assert fi["data"].attrs["env_args"] == fo["data"].attrs["env_args"]
        for new, old in enumerate(ids):
            a, b = fi["data"][f"demo_{old}"], fo["data"][f"demo_{new}"]
            assert np.array_equal(a["actions"][:], b["actions"][:])
            for key in a["obs"]:
                assert np.array_equal(a["obs"][key][:], b["obs"][key][:])


def main() -> None:
    src, out, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(src, "r") as f:
        n_total = len(f["data"])
    for seed in SEEDS:
        perm = np.random.default_rng([seed, 7]).permutation(n_total).tolist()
        for n in SIZES:
            ids = perm[:n]
            dst = out / f"{name}_N{n}_s{seed}.hdf5"
            write_subset(src, dst, ids)
            verify(src, dst, ids)
            dst.with_suffix(".json").write_text(json.dumps({"source": str(src), "seed": seed, "demo_ids": ids}))
            print(dst, "ok", flush=True)


if __name__ == "__main__":
    main()
