"""Stream the MICrONS synapse table and keep the synapses between the selected neurons.

    python pipeline/synapses.py

Reads pipeline/cache/selection.json (written by fetch.py), streams the 20 GB gzip table with
ranged HTTP reads of 64 MiB, inflates it block by block, and keeps every row whose pre AND post
root ids are both selected. Rows land in pipeline/cache/synapses_selected.csv:

    id,pre_root_id,post_root_id,ctr_x_vox,ctr_y_vox,ctr_z_vox,size

Coordinates are the source's 4x4x40 nm voxels, untouched; pack.py converts them.

Resuming: a deflate stream cannot be entered mid-way (the decompressor's bit state and 32 KB window
are not exposed by zlib), so a checkpoint cannot store "the decompressor state". Instead
pipeline/cache/synapses_selected.json records the last fully written block, the output size at
that point and the running counts. A rerun re-downloads and inflates the blocks before the
checkpoint without filtering them, truncates the csv to the checkpointed size, and carries on, so
no row is ever written twice and the result is identical to an uninterrupted run. The replay costs
download time only; filtering (the slow part) is skipped.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.request
import zlib
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import common as c

NAME = "synapses_pni_2_v1_filtered_view"
URL = c.CELL_BASE + NAME + ".csv.gz"
HEADER_URL = c.CELL_BASE + NAME + "_header.csv"
HEADER_FILE = c.CACHE / (NAME + "_header.csv")
OUT = c.CACHE / "synapses_selected.csv"
STATE = c.CACHE / "synapses_selected.json"
OUT_HEADER = b"id,pre_root_id,post_root_id,ctr_x_vox,ctr_y_vox,ctr_z_vox,size\n"

BLOCK = 64 * 1024 * 1024
PREFETCH = 4  # ranged downloads in flight
WORKERS = max(2, min(8, (os.cpu_count() or 4) - 2))

EXPECTED_COLUMNS = {0: "id", 4: "ctr_pt_position_x", 5: "ctr_pt_position_y", 6: "ctr_pt_position_z",
                    10: "size", 12: "pre_pt_root_id", 14: "post_pt_root_id"}


def request(url: str, headers: dict[str, str] | None = None, method: str = "GET"):
    last: Exception | None = None
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers=headers or {}, method=method)
            with urllib.request.urlopen(req, timeout=120) as r:
                return dict(r.headers), (r.read() if method == "GET" else b"")
        except Exception as e:  # network hiccup: back off and retry
            last = e
            time.sleep(min(30, 2 * (attempt + 1)))
    raise RuntimeError(f"{url}: {last}")


def fetch_block(index: int, total: int) -> bytes:
    start = index * BLOCK
    end = min(total, start + BLOCK) - 1
    for _ in range(4):
        _, body = request(URL, {"Range": f"bytes={start}-{end}"})
        if len(body) == end - start + 1:
            return body
    raise RuntimeError(f"block {index}: short read ({len(body)} of {end - start + 1} bytes)")


def check_header() -> None:
    if not HEADER_FILE.exists():
        HEADER_FILE.write_bytes(request(HEADER_URL)[1])
    names = [line.split(",")[0] for line in HEADER_FILE.read_text().splitlines() if line.strip()]
    for col, name in EXPECTED_COLUMNS.items():
        if names[col] != name:
            raise SystemExit(f"header column {col} is {names[col]!r}, expected {name!r}")
    if len(names) != 15:
        raise SystemExit(f"header has {len(names)} columns, expected 15")


# ------------------------------------------------------------------ filtering (runs in workers)

_SELECTED: frozenset = frozenset()


def _init_worker(selected: frozenset) -> None:
    global _SELECTED
    _SELECTED = selected


def filter_chunk(chunk: bytes) -> tuple[int, int, bytes]:
    """(rows seen, rows kept, kept rows as csv). `chunk` holds whole lines. The root ids are the
    last columns (checked against the header), so a cheap rsplit decides before any full split."""
    sel = _SELECTED
    kept: list[bytes] = []
    seen = 0
    for line in chunk.split(b"\n"):
        if not line:
            continue
        seen += 1
        _, pre, _, post = line.rsplit(b",", 3)
        if pre in sel and post in sel:
            f = line.split(b",")
            kept.append(b",".join((f[0], pre, post, f[4], f[5], f[6], f[10])))
    return seen, len(kept), (b"\n".join(kept) + b"\n") if kept else b""


# ------------------------------------------------------------------ the stream


class Inflater:
    """gzip inflate across blocks, tolerating concatenated members."""

    def __init__(self) -> None:
        self.d = zlib.decompressobj(31)
        self.ended = False  # True when the last byte fed closed a gzip member

    def feed(self, data: bytes) -> bytes:
        out = []
        while data:
            self.ended = False
            out.append(self.d.decompress(data))
            if not self.d.eof:
                break
            data = self.d.unused_data
            self.d = zlib.decompressobj(31)
            self.ended = True
        return b"".join(out)


def fmt_dur(s: float) -> str:
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def save_state(state: dict) -> None:
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=0) + "\n")
    tmp.replace(STATE)


def main() -> int:
    selected_ids = sorted(json.loads(c.SELECTION.read_text())["rootIds"])
    sha = hashlib.sha256(json.dumps(selected_ids).encode()).hexdigest()
    selected = frozenset(str(r).encode() for r in selected_ids)
    check_header()

    headers, _ = request(URL, method="HEAD")
    total = int(headers.get("Content-Length") or headers["content-length"])
    etag = headers.get("ETag", "")
    blocks = -(-total // BLOCK)

    state = None
    if STATE.exists():
        old = json.loads(STATE.read_text())
        if old["selectionSha"] == sha and old["etag"] == etag and old["gzBytes"] == total:
            state = old
    if state is not None and state["done"]:
        print(f"already complete: {state['rowsKept']:,} synapses kept of {state['rowsSeen']:,} rows "
              f"(delete {STATE.name} to redo)")
        return 0
    if state is not None and state["nextBlock"] >= blocks and OUT.exists() and OUT.stat().st_size == state["outBytes"]:
        state["done"] = True  # interrupted after the last block was written, before the flag was
        save_state(state)
        print(f"already complete: {state['rowsKept']:,} synapses kept")
        return 0
    if state is not None and (not OUT.exists() or OUT.stat().st_size < state["outBytes"]):
        state = None
    if state is None:
        state = {"selectionSha": sha, "rootIds": selected_ids, "etag": etag, "gzBytes": total,
                 "nextBlock": 0, "outBytes": len(OUT_HEADER), "rowsSeen": 0, "rowsKept": 0,
                 "liveSeconds": 0.0, "liveBytes": 0, "done": False}
        OUT.write_bytes(OUT_HEADER)
    else:
        os.truncate(OUT, state["outBytes"])
        print(f"resuming at block {state['nextBlock']}/{blocks}: replaying {state['nextBlock'] * BLOCK / 1e9:.1f} GB "
              f"without filtering, {state['rowsKept']:,} rows already kept")

    resume_at = state["nextBlock"]
    print(f"{NAME}.csv.gz: {total / 1e9:.2f} GB in {blocks} blocks of {BLOCK >> 20} MiB; "
          f"selecting among {len(selected_ids)} neurons; {WORKERS} filter workers")

    out = OUT.open("ab")
    inflater = Inflater()
    carry = b""
    inflight: deque = deque()  # (block index, future), in block order
    t_start = time.time()
    live_start_bytes = min(total, resume_at * BLOCK)
    live_seconds_before = state["liveSeconds"]
    live_bytes_before = state["liveBytes"]
    t_live = [t_start if resume_at == 0 else None]  # set once the replay is over

    def retire(block: int, future) -> None:
        seen, kept, data = future.result()
        out.write(data)
        out.flush()
        state["rowsSeen"] += seen
        state["rowsKept"] += kept
        state["outBytes"] = out.tell()
        state["nextBlock"] = block + 1
        done_bytes = min(total, (block + 1) * BLOCK)
        elapsed = time.time() - t_live[0]
        state["liveSeconds"] = live_seconds_before + elapsed
        state["liveBytes"] = live_bytes_before + done_bytes - live_start_bytes
        save_state(state)
        rate = (done_bytes - live_start_bytes) / max(elapsed, 1e-9)
        eta = (total - done_bytes) / max(rate, 1e-9)
        print(f"  block {block + 1}/{blocks}: {done_bytes / 1e9:6.2f}/{total / 1e9:.2f} GB "
              f"({100 * done_bytes / total:4.1f}%), {state['rowsSeen'] / 1e6:6.1f}M rows seen, "
              f"{state['rowsKept']:,} kept, {rate / 1e6:5.1f} MB/s, ETA {fmt_dur(eta)}", flush=True)

    with ThreadPoolExecutor(PREFETCH) as downloads, \
            ProcessPoolExecutor(WORKERS, initializer=_init_worker, initargs=(selected,)) as filters:
        pending = {b: downloads.submit(fetch_block, b, total) for b in range(resume_at, min(blocks, resume_at + PREFETCH))}
        # Replayed blocks are downloaded on demand, not through the prefetch window.
        for b in range(blocks):
            if b < resume_at:
                raw = inflater.feed(fetch_block(b, total))
                carry = (carry + raw)[(carry + raw).rfind(b"\n") + 1:]
                continue
            if t_live[0] is None:
                t_live[0] = time.time()
            data = pending.pop(b).result()
            nxt = b + PREFETCH
            if nxt < blocks:
                pending[nxt] = downloads.submit(fetch_block, nxt, total)
            text = carry + inflater.feed(data)
            cut = text.rfind(b"\n") + 1
            chunk, carry = text[:cut], text[cut:]
            if b == blocks - 1:
                if not inflater.ended:
                    raise RuntimeError("the gzip stream ended before its end marker")
                chunk += carry + (b"\n" if carry else b"")
                carry = b""
            inflight.append((b, filters.submit(filter_chunk, chunk)))
            del text, chunk
            while inflight and (len(inflight) > WORKERS or inflight[0][1].done()):
                retire(*inflight.popleft())
        while inflight:
            retire(*inflight.popleft())
    out.close()

    state["done"] = True
    save_state(state)
    rate = state["liveBytes"] / max(state["liveSeconds"], 1e-9)
    print(f"done: {state['rowsKept']:,} synapses kept of {state['rowsSeen']:,} rows; "
          f"{state['liveBytes'] / 1e9:.2f} GB streamed in {fmt_dur(state['liveSeconds'])} "
          f"(average {rate / 1e6:.1f} MB/s); wall time of this run {fmt_dur(time.time() - t_start)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
