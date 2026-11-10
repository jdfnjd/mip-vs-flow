"""Rebuild the robosuite commit pinned in uv.lock from the PyPI wheel plus single files.

GitHub tarballs and clones keep dropping on this server. The wheel holds nearly the
same files, so only files whose git hash differs are fetched, and every file is
checked against the commit's git tree.

Usage: python rebuild_robosuite.py <wheel> <out_dir>   then: python rebuild_robosuite.py --verify <site_pkgs>
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request as U
import zipfile
from pathlib import Path

SHA = "a071383d53568ab798eb315c0e95357911be922d"
REPO = "ARISE-Initiative/robosuite"


def get(url: str) -> bytes:
    for k in range(10):
        try:
            return U.urlopen(U.Request(url, headers={"User-Agent": "mip-density"}), timeout=60).read()
        except OSError:
            if k == 9:
                raise
            time.sleep(10)
    raise AssertionError


def git_hash(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def tree() -> dict[str, str]:
    t = json.loads(get(f"https://api.github.com/repos/{REPO}/git/trees/{SHA}?recursive=1"))
    assert not t["truncated"]
    return {e["path"]: e["sha"] for e in t["tree"] if e["type"] == "blob"}


def rebuild(wheel: str, out: str) -> None:
    blobs = tree()
    z = zipfile.ZipFile(wheel)
    names = set(z.namelist())
    wanted = [p for p in blobs if p.startswith("robosuite/") or "/" not in p]
    fetched = 0
    for p in wanted:
        b = z.read(p) if p in names else None
        if b is None or git_hash(b) != blobs[p]:
            b = get(f"https://raw.githubusercontent.com/{REPO}/{SHA}/{p}")
            fetched += 1
        assert git_hash(b) == blobs[p], p
        f = Path(out) / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b)
    print(f"rebuilt {len(wanted)} files, {fetched} fetched, all match commit {SHA}")


def verify(site: str) -> None:
    blobs = {p: s for p, s in tree().items() if p.startswith("robosuite/")}
    bad = [p for p, s in blobs.items() if not (Path(site) / p).exists() or git_hash((Path(site) / p).read_bytes()) != s]
    print(f"installed: {len(blobs) - len(bad)}/{len(blobs)} files match commit {SHA}")
    print("not matching:", bad)


if __name__ == "__main__":
    verify(sys.argv[2]) if sys.argv[1] == "--verify" else rebuild(sys.argv[1], sys.argv[2])
