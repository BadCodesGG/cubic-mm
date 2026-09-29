"""Download the MICrONS proofread skeletons for the neurons the site shows.

    python pipeline/fetch.py --count 200      # the 200 somas nearest the centroid
    python pipeline/fetch.py --count all      # every matched proofread skeleton

Everything lands in pipeline/cache/ (gitignored). Re-running skips files that are already there
with the size the bucket listing reports, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import common as c


def get(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception as e:  # network hiccup: back off and retry
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{url}: {last}")


def list_bucket() -> list[dict]:
    items: list[dict] = []
    token = None
    while True:
        url = c.LISTING_URL + (f"&pageToken={urllib.parse.quote(token)}" if token else "")
        page = json.loads(get(url))
        items += page.get("items", [])
        token = page.get("nextPageToken")
        if not token:
            break
    return sorted((i for i in items if i["name"].endswith(".swc")), key=lambda i: i["name"])


def ensure_file(path, url: str) -> None:
    if not path.exists():
        path.write_bytes(get(url))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", default="200", help="a number, or 'all'")
    args = ap.parse_args()

    c.SWC_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    items = list_bucket()
    c.LISTING.write_text(json.dumps(items, indent=0) + "\n")
    print(f"bucket listing: {len(items)} skeletons, {sum(int(i['size']) for i in items) / 1e6:.0f} MB")

    ensure_file(c.CELL_TABLE, c.CELL_BASE + "aibs_cell_info.csv.gz")
    ensure_file(c.CELL_HEADER, c.CELL_BASE + "aibs_cell_info_header.csv")
    cells = c.load_cells()
    listing = c.load_listing()
    ids = c.matched(listing, cells)
    print(f"cell table: {len(cells)} cells; {len(ids)} of {len(listing)} proofread ids matched, {len(listing) - len(ids)} skipped")

    somas = [cells[r]["somaUm"] for r in ids]
    centroid = tuple(sum(s[k] for s in somas) / len(somas) for k in range(3))
    dist = {r: math.dist(cells[r]["somaUm"], centroid) for r in ids}
    order = sorted(ids, key=lambda r: (dist[r], r))
    chosen = order if args.count == "all" else order[: int(args.count)]
    print(f"centroid (um): {centroid[0]:.1f} {centroid[1]:.1f} {centroid[2]:.1f}; selected {len(chosen)}, "
          f"farthest {dist[chosen[-1]]:.1f} um from it")

    sizes = {int(i["name"].rsplit("/", 1)[1][:-4]): int(i["size"]) for i in items}
    todo = [r for r in chosen if not (c.SWC_DIR / f"{r}.swc").exists()
            or (c.SWC_DIR / f"{r}.swc").stat().st_size != sizes[r]]
    print(f"downloading {len(todo)} skeletons ({sum(sizes[r] for r in todo) / 1e6:.0f} MB), "
          f"{len(chosen) - len(todo)} already cached")

    def fetch_one(r: int) -> None:
        tmp = c.SWC_DIR / f"{r}.swc.part"
        tmp.write_bytes(get(c.SWC_URL.format(root_id=r)))
        tmp.replace(c.SWC_DIR / f"{r}.swc")

    with ThreadPoolExecutor(max_workers=8) as pool:
        for n, _ in enumerate(pool.map(fetch_one, todo), 1):
            if n % 25 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}")

    c.SELECTION.write_text(json.dumps({"count": len(chosen), "rootIds": chosen}, indent=0) + "\n")
    print(f"done in {time.time() - t0:.1f}s; selection written to {c.SELECTION.relative_to(c.ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
