"""Download + verify + unpack Carinthia-S (Zenodo 10.5281/zenodo.16895427, CC BY 4.0, 133 MB).

python scripts/download_data.py [--dest data/raw/carinthia_s]
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "data.zip": (
        "https://zenodo.org/api/records/16895427/files/data.zip/content",
        "3a41242574635cf1302fab15accb7730",
    ),
    "carinthia-s_dataset.html": (
        "https://zenodo.org/api/records/16895427/files/carinthia-s_dataset.html/content",
        "4d9ca3ed51dfc97444939f742496d2d9",
    ),
}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="data/raw/carinthia_s")
    dest = ROOT / ap.parse_args().dest
    dest.mkdir(parents=True, exist_ok=True)
    for name, (url, digest) in FILES.items():
        path = dest / name
        if not path.exists() or md5(path) != digest:
            print(f"downloading {name} ...", flush=True)
            urllib.request.urlretrieve(url, path)
        got = md5(path)
        if got != digest:
            raise SystemExit(f"checksum mismatch for {name}: {got} != {digest}")
        print(f"{name}: md5 OK")
    if not (dest / "data" / "carinthia-s.csv").exists():
        with zipfile.ZipFile(dest / "data.zip") as z:
            z.extractall(dest)
    n = len(list((dest / "data" / "images").glob("*.jpg")))
    print(f"unpacked {n} images to {dest / 'data'}")


if __name__ == "__main__":
    main()
