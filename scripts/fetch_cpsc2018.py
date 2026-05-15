"""Mirror PhysioNet's challenge-2020/training/cpsc_2018/ subdirectory.

Equivalent to:
    wget -c -r -N -np https://physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/

Implementation:
  - Recurse the HTML directory index (g1/.../g7/) and collect .mat + .hea
  - Parallel download with ThreadPoolExecutor (default 16 workers)
  - Skip files already on disk that match remote size; otherwise resume
  - File names are flattened (CPSC IDs A0001.mat etc. are globally unique)

Typical run: ~1.4 GB across ~13700 files (6877 records × {.mat, .hea}).
With 16 workers usually finishes in 5-10 min.
"""
from __future__ import annotations

import argparse
import re
import sys
import threading
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

INDEX_URL = (
    "https://physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/"
)


def fetch(url: str, retries: int = 5, timeout: float = 60.0) -> bytes:
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"failed after {retries} retries: {last}")


def list_files(index_url: str) -> list[tuple[str, str]]:
    """Recursively walk the PhysioNet directory index.

    Returns list of (full_url, basename) for every .mat / .hea found.
    """
    out: list[tuple[str, str]] = []

    def walk(url: str):
        html = fetch(url).decode("utf-8", errors="replace")
        for m in re.finditer(r'href="([^"]+)"', html):
            href = m.group(1)
            if href in ("../", "./") or href.startswith("?") or href.startswith("#"):
                continue
            if href.endswith("/"):
                walk(url + href)
            elif href.endswith(".mat") or href.endswith(".hea"):
                out.append((url + href, href.split("/")[-1]))
    walk(index_url)
    out.sort(key=lambda t: t[1])
    return out


def download_one(url: str, out: Path, retries: int = 4) -> tuple[bool, int]:
    """Download `url` to `out` with size-check + resume. Return (downloaded, bytes_now)."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            # HEAD for remote size
            req = urllib.request.Request(
                url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                total = int(r.headers.get("Content-Length", "0"))

            cur = out.stat().st_size if out.exists() else 0
            if total and cur >= total:
                return False, cur

            headers = {"User-Agent": "Mozilla/5.0"}
            mode = "wb"
            if cur > 0 and total > cur:
                headers["Range"] = f"bytes={cur}-"
                mode = "ab"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r, open(out, mode) as f:
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    f.write(buf)
            return True, out.stat().st_size
        except Exception as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"download failed for {url}: {last_err}")


_print_lock = threading.Lock()
_progress = {"done": 0, "dl": 0, "skip": 0, "bytes": 0, "total": 0}


def _worker(item, out_dir):
    url, name = item
    out = out_dir / name
    try:
        downloaded, sz = download_one(url, out)
    except Exception as e:
        with _print_lock:
            print(f"  [ERR] {name}: {e}")
        return
    with _print_lock:
        _progress["done"] += 1
        _progress["bytes"] += sz if downloaded else 0
        if downloaded:
            _progress["dl"] += 1
        else:
            _progress["skip"] += 1
        if _progress["done"] % 200 == 0 or _progress["done"] == _progress["total"]:
            elapsed = time.time() - _progress["t0"]
            rate = _progress["bytes"] / max(elapsed, 1e-6) / 1e6
            print(f"  [{_progress['done']}/{_progress['total']}] "
                  f"dl={_progress['dl']} skip={_progress['skip']} "
                  f"new_bytes={_progress['bytes']/1e6:.0f} MB "
                  f"avg={rate:.1f} MB/s "
                  f"elapsed={elapsed/60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"c:\ZJU\ECG\data\cpsc2018_raw")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mat-only", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"index:   {INDEX_URL}")
    print(f"out:     {out_dir}")
    print(f"workers: {args.workers}")
    print("listing remote directory ...", flush=True)
    items = list_files(INDEX_URL)
    if args.mat_only:
        items = [t for t in items if t[1].endswith(".mat")]
    if args.limit:
        items = items[: args.limit]
    print(f"  files to consider: {len(items)}", flush=True)

    _progress["total"] = len(items)
    _progress["t0"] = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_worker, it, out_dir) for it in items]
        for _ in as_completed(futures):
            pass

    elapsed = (time.time() - _progress["t0"]) / 60
    print(f"\ndone: {_progress['dl']} downloaded, {_progress['skip']} skipped, "
          f"{_progress['bytes']/1e6:.0f} MB new, {elapsed:.1f} min", flush=True)


if __name__ == "__main__":
    main()
