"""Shared paths and the cell-table join used by fetch.py and pack.py."""

from __future__ import annotations

import csv
import gzip
import json
import struct
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent
ROOT = PIPELINE.parent
CACHE = PIPELINE / "cache"
SWC_DIR = CACHE / "swc"
OUT = ROOT / "public" / "data"

BUCKET = "microns-static-links"
SWC_PREFIX = "skel/swc/proofread/"
LISTING_URL = (
    "https://storage.googleapis.com/storage/v1/b/" + BUCKET + "/o"
    "?prefix=" + SWC_PREFIX + "&fields=items(name,size),nextPageToken"
)
SWC_URL = "https://storage.googleapis.com/" + BUCKET + "/" + SWC_PREFIX + "{root_id}.swc"
CELL_BASE = "https://storage.googleapis.com/mat_dbs/public/minnie65_phase3_v1/v1300/"
CELL_TABLE = CACHE / "aibs_cell_info.csv.gz"
CELL_HEADER = CACHE / "aibs_cell_info_header.csv"
LISTING = CACHE / "proofread_listing.json"
SELECTION = CACHE / "selection.json"

# EM info bounds in nm (min, max), used as the global quantisation bounds.
BOUNDS_NM = ([110592, 110592, 592640], [1814528, 1552384, 1116160])
BOUNDS_UM = ([v / 1000 for v in BOUNDS_NM[0]], [v / 1000 for v in BOUNDS_NM[1]])


def decode_position_um(ewkb_hex: str) -> tuple[float, float, float]:
    """EWKB point (little-endian, Z flag, no SRID) in 4x4x40 nm voxels -> micrometres."""
    x, y, z = struct.unpack("<ddd", bytes.fromhex(ewkb_hex)[5:29])
    return (x * 4 / 1000, y * 4 / 1000, z * 40 / 1000)


def load_listing() -> list[int]:
    """Sorted root ids of every proofread skeleton in the bucket."""
    items = json.loads(LISTING.read_text())
    return sorted(int(Path(i["name"]).stem) for i in items)


def load_cells() -> dict[int, dict]:
    """pt_root_id -> {somaUm, broad_type, cell_type}. The first row wins on a duplicate id."""
    names = [line.split(",")[0] for line in CELL_HEADER.read_text().splitlines() if line.strip()]
    cells: dict[int, dict] = {}
    with gzip.open(CELL_TABLE, "rt", newline="") as f:
        for row in csv.reader(f):
            rec = dict(zip(names, row))
            rid = int(rec["pt_root_id"])
            if rid in cells or not rec["pt_position"]:
                continue
            cells[rid] = {
                "somaUm": decode_position_um(rec["pt_position"]),
                "broad_type": rec["broad_type"],
                "cell_type": rec["cell_type"],
            }
    return cells


def matched(listing: list[int], cells: dict[int, dict]) -> list[int]:
    return [rid for rid in listing if rid in cells]
