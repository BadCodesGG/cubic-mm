"""Pack the cached MICrONS skeletons and synapses into the binary layout in src/engine/format.ts.

    python pipeline/pack.py            # write public/data/
    python pipeline/pack.py --check    # re-pack in memory, exit 1 if public/data/ differs

The output is a pure function of pipeline/cache/: fixed sort orders, no timestamps. Skeletons are
parsed and decimated in worker processes, but results are gathered in a fixed order.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import common as c

# Per-LOD (Ramer-Douglas-Peucker tolerance, maximum edge length, shortest terminal twig kept), all in
# micrometres. Tuned so the whole set stays under NODE_BUDGET; the totals are printed on every pack.
LODS = {"hi": (2.0, 28.0, 0.0), "lite": (8.0, 150.0, 50.0)}
NODE_BUDGET = {"hi": 1_200_000, "lite": 350_000}
FULL_RES_MAX_EDGE_UM = 12.0
CHUNK_TARGET_BYTES = 1_500_000
BYTES_PER_NODE = 4 + 6 + 2 + 2 + 1  # parent, pos, radius, pathDist, compartment
NO_PARENT = 0xFFFFFFFF
MAX_NODES = 65_535

# A synapse further than this from every full-resolution axon node of its presynaptic neuron takes
# the path distance of the nearest node of any compartment instead.
AXON_MATCH_UM = 5.0
SYNAPSE_VOXEL_NM = (4, 4, 40)

SOMA, AXON, DENDRITE = 0, 1, 2
SWC_TYPE_TO_COMPARTMENT = {1: SOMA, 2: AXON, 3: DENDRITE, 4: DENDRITE}
EXCITATORY_LAYERS = {"23P": 2, "4P": 4, "5P": 5, "6P": 6}

CREDITS = {
    "dataset": "MICrONS minnie65 (cubic millimetre of mouse visual cortex), proofread skeletons v1300 and aibs_cell_info",
    "licence": "CC BY 4.0",
    "url": "https://www.microns-explorer.org/cortical-mm3",
    "citations": [
        "The MICrONS Consortium. Functional connectomics spanning multiple areas of mouse visual cortex. "
        "Nature 640, 435-447 (2025). https://doi.org/10.1038/s41586-025-08790-w"
    ],
}


# ------------------------------------------------------------------ binary writer (mirrors format.ts)


def _align(n: int, a: int) -> int:
    return (n + a - 1) // a * a


class Writer:
    """Lays arrays out the way format.ts's Writer does: each section aligned, zero padded."""

    def __init__(self, start: int) -> None:
        self.offset = start
        self.parts: list[tuple[int, bytes]] = []

    def put(self, arr: np.ndarray, dtype: str, alignment: int) -> None:
        raw = np.ascontiguousarray(arr, dtype=dtype).tobytes()
        self.offset = _align(self.offset, alignment)
        self.parts.append((self.offset, raw))
        self.offset += len(raw)

    def u64(self, a: np.ndarray) -> None:
        self.put(a, "<u8", 8)

    def u32(self, a: np.ndarray) -> None:
        self.put(a, "<u4", 4)

    def f32(self, a: np.ndarray) -> None:
        self.put(a, "<f4", 4)

    def u16(self, a: np.ndarray) -> None:
        self.put(a, "<u2", 4)

    def u8(self, a: np.ndarray) -> None:
        self.put(a, "u1", 4)

    def finish(self, header: bytes) -> bytes:
        out = bytearray(_align(self.offset, 4))
        out[: len(header)] = header
        for at, raw in self.parts:
            out[at : at + len(raw)] = raw
        return bytes(out)


def u32le(*vals: int) -> bytes:
    return b"".join(int(v).to_bytes(4, "little") for v in vals)


# ------------------------------------------------------------------ skeleton parsing


class Skeleton:
    """One neuron at full resolution, soma first, every parent before its children."""

    def __init__(self, pos, radius, parent, comp, dist):
        self.pos = pos  # (n, 3) um
        self.radius = radius  # (n,) um
        self.parent = parent  # (n,) int, -1 for the soma
        self.comp = comp  # (n,) uint8
        self.dist = dist  # (n,) um along the tree from the soma


def parse_swc(path, root_id: int) -> Skeleton:
    a = np.loadtxt(path, ndmin=2)
    ids = a[:, 0].astype(np.int64)
    index_of = {int(i): k for k, i in enumerate(ids)}
    par = np.array([index_of.get(int(p), -1) for p in a[:, 6]], dtype=np.int64)
    roots = np.flatnonzero(par < 0)
    if len(roots) != 1 or int(a[roots[0], 1]) != 1:
        raise ValueError(f"{root_id}: expected one soma root, got {len(roots)} roots")
    pos = a[:, 2:5].astype(np.float64)
    rad = a[:, 5].astype(np.float64)
    types = a[:, 1].astype(np.int64)
    bad = set(types.tolist()) - set(SWC_TYPE_TO_COMPARTMENT)
    if bad:
        raise ValueError(f"{root_id}: unknown SWC types {sorted(bad)}")
    comp = np.array([SWC_TYPE_TO_COMPARTMENT[t] for t in types], dtype=np.uint8)

    pos, rad, par, comp = subdivide_long_edges(pos, rad, par, comp)

    # Depth-first preorder from the soma, children in file order: parents precede children and
    # each branch stays contiguous.
    n = len(pos)
    child_order = np.argsort(par, kind="stable")
    sorted_par = par[child_order]
    starts = np.searchsorted(sorted_par, np.arange(n + 1), side="left")  # runs per parent
    order = []
    stack = [int(roots[0])]
    while stack:
        i = stack.pop()
        order.append(i)
        kids = child_order[starts[i] : starts[i + 1]]
        stack.extend(int(k) for k in kids[::-1])
    if len(order) != n:
        raise ValueError(f"{root_id}: {n - len(order)} nodes are not connected to the soma")
    order_arr = np.array(order, dtype=np.int64)
    new_of = np.empty(n, dtype=np.int64)
    new_of[order_arr] = np.arange(n)
    pos, rad, comp = pos[order_arr], rad[order_arr], comp[order_arr]
    par = np.where(par[order_arr] < 0, -1, new_of[np.maximum(par[order_arr], 0)])

    # Path length from the soma at full resolution, before any decimation.
    edge = np.zeros(n)
    edge[1:] = np.linalg.norm(pos[1:] - pos[par[1:]], axis=1)
    dist = np.zeros(n)
    for i in range(1, n):
        dist[i] = dist[par[i]] + edge[i]
    return Skeleton(pos, rad, par, comp, dist)


def subdivide_long_edges(pos, rad, par, comp):
    """Insert evenly spaced nodes into any original edge longer than FULL_RES_MAX_EDGE_UM, so the
    full-res skeleton has none. The new nodes take the child's compartment."""
    has_parent = par >= 0
    edge = np.zeros(len(pos))
    edge[has_parent] = np.linalg.norm(pos[has_parent] - pos[par[has_parent]], axis=1)
    long_idx = np.flatnonzero(edge > FULL_RES_MAX_EDGE_UM)
    if len(long_idx) == 0:
        return pos, rad, par, comp
    new_pos, new_rad, new_comp = [], [], []
    par = par.copy()
    next_id = len(pos)
    for i in long_idx:
        p = par[i]
        pieces = int(math.ceil(edge[i] / FULL_RES_MAX_EDGE_UM))
        prev = p
        for k in range(1, pieces):
            t = k / pieces
            new_pos.append(pos[p] + t * (pos[i] - pos[p]))
            new_rad.append(rad[p] + t * (rad[i] - rad[p]))
            new_comp.append(comp[i])
            par = np.append(par, prev)
            prev = next_id
            next_id += 1
        par[i] = prev
    return (
        np.vstack([pos, np.array(new_pos)]),
        np.concatenate([rad, np.array(new_rad)]),
        par,
        np.concatenate([comp, np.array(new_comp, dtype=np.uint8)]),
    )


# ------------------------------------------------------------------ decimation


def rdp_interior(points: np.ndarray, tol: float, max_len: float) -> list[int]:
    """Indices strictly inside `points` that must be kept: Ramer-Douglas-Peucker, plus a split
    wherever the chord between kept points is longer than max_len."""
    kept: list[int] = []
    stack = [(0, len(points) - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2:
            continue
        chord = points[hi] - points[lo]
        length = float(np.linalg.norm(chord))
        rel = points[lo + 1 : hi] - points[lo]
        if length > 1e-12:
            t = np.clip(rel @ chord / (length * length), 0.0, 1.0)
            dev = np.linalg.norm(rel - t[:, None] * chord, axis=1)
        else:
            dev = np.linalg.norm(rel, axis=1)
        k = int(np.argmax(dev))
        if dev[k] > tol:
            split = lo + 1 + k
        elif length > max_len:
            split = lo + (hi - lo) // 2
        else:
            continue
        kept.append(split)
        stack.append((lo, split))
        stack.append((split, hi))
    return kept


class Decimated:
    def __init__(self, pos, radius, parent, comp, dist):
        self.pos, self.radius, self.parent, self.comp, self.dist = pos, radius, parent, comp, dist

    @property
    def count(self) -> int:
        return len(self.pos)


def prune_twigs(sk: Skeleton, min_len: float) -> Skeleton:
    """Drop every terminal branch (tip up to its branch point) shorter than `min_len`. One pass, so
    a branch point left with a single child simply stops being a key node. Twigs that would take a
    neurite all the way back to the soma are kept."""
    if min_len <= 0:
        return sk
    n = len(sk.pos)
    par = sk.parent
    child_count = np.bincount(par[1:], minlength=n)
    edge = np.zeros(n)
    edge[1:] = np.linalg.norm(sk.pos[1:] - sk.pos[par[1:]], axis=1)
    drop = np.zeros(n, dtype=bool)
    for tip in np.flatnonzero(child_count == 0):
        run, length, cur = [], 0.0, int(tip)
        while cur != 0 and child_count[cur] <= 1:
            run.append(cur)
            length += edge[cur]
            cur = int(par[cur])
        if cur != 0 and length < min_len:
            drop[run] = True
    keep = ~drop
    rank = np.cumsum(keep) - 1
    new_parent = np.full(int(keep.sum()), -1, dtype=np.int64)
    new_parent[1:] = rank[par[keep][1:]]
    return Skeleton(sk.pos[keep], sk.radius[keep], new_parent, sk.comp[keep], sk.dist[keep])


def decimate(sk: Skeleton, tol: float, max_len: float) -> Decimated:
    n = len(sk.pos)
    par = sk.parent
    child_count = np.bincount(par[1:], minlength=n)
    # The one child of every node that has exactly one (used to walk unbranched runs).
    only_child = np.full(n, -1, dtype=np.int64)
    only_child[par[1:]] = np.arange(1, n)  # correct wherever child_count == 1

    key = (child_count != 1) | np.concatenate([[True], sk.comp[1:] != sk.comp[par[1:]]])
    key[0] = True
    keep = key.copy()

    # Children of every key node, in index order.
    order = np.argsort(par[1:], kind="stable") + 1
    starts = np.searchsorted(par[order], np.arange(n + 1), side="left")
    for k in np.flatnonzero(key):
        for ch in order[starts[k] : starts[k + 1]]:
            run = [int(k), int(ch)]
            cur = int(ch)
            while not key[cur]:
                cur = int(only_child[cur])
                run.append(cur)
            if len(run) > 2:
                for j in rdp_interior(sk.pos[run], tol, max_len):
                    keep[run[j]] = True

    kept_idx = np.flatnonzero(keep)
    nearest_kept = np.empty(n, dtype=np.int64)
    for i in range(n):
        nearest_kept[i] = i if keep[i] else nearest_kept[par[i]]
    rank = np.cumsum(keep) - 1
    new_parent = np.full(len(kept_idx), -1, dtype=np.int64)
    new_parent[1:] = rank[nearest_kept[par[kept_idx[1:]]]]
    return Decimated(sk.pos[kept_idx], sk.radius[kept_idx], new_parent, sk.comp[kept_idx], sk.dist[kept_idx])


# ------------------------------------------------------------------ quantisation and ordering


def quantise(values_um: np.ndarray, lo: list[float], hi: list[float]) -> np.ndarray:
    lo_a, hi_a = np.array(lo), np.array(hi)
    if (values_um < lo_a).any() or (values_um > hi_a).any():
        raise ValueError("a node lies outside the global bounds")
    t = (values_um - lo_a) / (hi_a - lo_a)
    return np.floor(t * 65535 + 0.5).astype(np.uint16)  # JS Math.round: half rounds up


def dequantise(q: np.ndarray, lo: list[float], hi: list[float]) -> np.ndarray:
    lo_a, hi_a = np.array(lo), np.array(hi)
    return lo_a + (q.astype(np.float64) / 65535) * (hi_a - lo_a)


def morton(qx: int, qy: int, qz: int) -> int:
    code = 0
    for bit in range(16):
        code |= ((qx >> bit) & 1) << (3 * bit)
        code |= ((qy >> bit) & 1) << (3 * bit + 1)
        code |= ((qz >> bit) & 1) << (3 * bit + 2)
    return code


def layer_bands(cells: dict, ids: list[int]) -> list[float]:
    """y boundaries between layers 1|2, 2|4, 4|5, 5|6, from the excitatory cells of every matched
    proofread neuron. Pia is at low y, so layers stack with increasing y."""
    ys: dict[int, list[float]] = {2: [], 4: [], 5: [], 6: []}
    for r in ids:
        cell = cells[r]
        layer = excitatory_layer(cell)
        if layer:
            ys[layer].append(cell["somaUm"][1])
    q = {layer: np.array(v) for layer, v in ys.items()}
    return [
        float(np.quantile(q[2], 0.05)),
        float((np.quantile(q[2], 0.9) + np.quantile(q[4], 0.1)) / 2),
        float((np.quantile(q[4], 0.9) + np.quantile(q[5], 0.1)) / 2),
        float((np.quantile(q[5], 0.9) + np.quantile(q[6], 0.1)) / 2),
    ]


def excitatory_layer(cell: dict) -> int:
    if cell["broad_type"] != "excitatory":
        return 0
    return EXCITATORY_LAYERS.get(cell["cell_type"].split("-")[0], 0)


def layer_of(cell: dict, bands: list[float]) -> int:
    layer = excitatory_layer(cell)
    if layer:
        return layer
    y = cell["somaUm"][1]
    return [1, 2, 4, 5, 6][sum(y >= b for b in bands)]


# ------------------------------------------------------------------ synapses


def load_synapse_rows(selected: list[int]) -> dict[str, np.ndarray] | None:
    """Rows of pipeline/cache/synapses_selected.csv whose pre and post are both in `selected`, or
    None if synapses.py has not produced a complete file for (a superset of) this selection."""
    if not (c.SYNAPSES_CSV.exists() and c.SYNAPSES_STATE.exists()):
        return None
    state = json.loads(c.SYNAPSES_STATE.read_text())
    if not state["done"]:
        return None
    if not set(selected) <= set(state["rootIds"]):
        raise ValueError("synapses_selected.csv was built for a different selection; rerun synapses.py")
    keep = set(selected)
    ids, pre, post, xyz, size = [], [], [], [], []
    with c.SYNAPSES_CSV.open(newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            a, b = int(row[1]), int(row[2])
            if a in keep and b in keep:
                ids.append(int(row[0]))
                pre.append(a)
                post.append(b)
                xyz.append((float(row[3]), float(row[4]), float(row[5])))
                size.append(float(row[6]))
    return {
        "id": np.array(ids, dtype=np.int64),
        "pre": np.array(pre, dtype=np.int64),
        "post": np.array(post, dtype=np.int64),
        "um": np.array(xyz, dtype=np.float64).reshape(-1, 3) * (np.array(SYNAPSE_VOXEL_NM) / 1000.0),
        "size": np.array(size, dtype=np.float64),
    }


def nearest_node(points: np.ndarray, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index into `nodes` of the closest node to each point, and that distance. Plain broadcasting
    rather than a matmul trick, so ties resolve the same way on every machine."""
    idx = np.empty(len(points), dtype=np.int64)
    dist = np.empty(len(points))
    step = max(1, 1_000_000 // len(nodes))
    for i in range(0, len(points), step):
        d2 = ((points[i : i + step, None, :] - nodes[None, :, :]) ** 2).sum(axis=2)
        j = d2.argmin(axis=1)
        idx[i : i + step] = j
        dist[i : i + step] = np.sqrt(d2[np.arange(len(j)), j])
    return idx, dist


def pre_path_dist(sk: Skeleton, synapse_um: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Path length from the soma (um) to the full-res axon node nearest each synapse, and a mask of
    the synapses that had no axon node within AXON_MATCH_UM and used the nearest node of any
    compartment instead."""
    axon = np.flatnonzero(sk.comp == AXON)
    idx, dist = nearest_node(synapse_um, sk.pos[axon])
    path = sk.dist[axon[idx]]
    far = dist > AXON_MATCH_UM
    if far.any():
        any_idx, _ = nearest_node(synapse_um[far], sk.pos)
        path[far] = sk.dist[any_idx]
    return path, far


# ------------------------------------------------------------------ per-neuron work (in workers)


def process_neuron(job: tuple[int, np.ndarray | None]) -> dict:
    """Parse one skeleton, decimate it per LOD and, when it has synapses, place them on its axon."""
    root_id, synapse_um = job
    sk = parse_swc(c.SWC_DIR / f"{root_id}.swc", root_id)
    out = {
        "root": root_id,
        "soma": sk.pos[0].copy(),
        "axonNodes": int((sk.comp == AXON).sum()),
        "decimated": {name: decimate(prune_twigs(sk, twig), tol, cap) for name, (tol, cap, twig) in LODS.items()},
        "preDist": None,
        "fallback": None,
    }
    if synapse_um is not None and len(synapse_um) and out["axonNodes"]:
        out["preDist"], out["fallback"] = pre_path_dist(sk, synapse_um)
    return out


# ------------------------------------------------------------------ the pack


def pack(log=print) -> dict[str, bytes]:
    cells = c.load_cells()
    listing = c.load_listing()
    all_matched = c.matched(listing, cells)
    bands = layer_bands(cells, all_matched)
    log(f"layer y bands (um, pia at low y): 1|2 {bands[0]:.0f}  2|4 {bands[1]:.0f}  4|5 {bands[2]:.0f}  5|6 {bands[3]:.0f}")

    selected = json.loads(c.SELECTION.read_text())["rootIds"]
    lo, hi = c.BOUNDS_UM

    rows = load_synapse_rows(selected)
    by_pre = {r: np.flatnonzero(rows["pre"] == r) for r in selected} if rows is not None else {}
    jobs = [(r, rows["um"][by_pre[r]] if rows is not None else None) for r in selected]
    with ProcessPoolExecutor(min(os.cpu_count() or 1, 12)) as pool:
        results = {res["root"]: res for res in pool.map(process_neuron, jobs, chunksize=8)}

    def soma_key(r: int) -> tuple[int, int]:
        q = quantise(results[r]["soma"][None, :], lo, hi)[0]
        return (morton(int(q[0]), int(q[1]), int(q[2])), r)

    order = sorted(selected, key=soma_key)
    n_neurons = len(order)

    cell_types = sorted({cells[r]["cell_type"] for r in order})
    type_index = {t: i for i, t in enumerate(cell_types)}
    if len(cell_types) > 255:
        raise ValueError("cellType does not fit in a byte")

    files: dict[str, bytes] = {}

    neurons = Writer(8)
    neurons.u64(np.array(order, dtype=np.uint64))
    neurons.f32(np.array([results[r]["soma"] for r in order], dtype=np.float32))
    neurons.u8(np.array([type_index[cells[r]["cell_type"]] for r in order]))
    neurons.u8(np.array([cells[r]["broad_type"] == "inhibitory" for r in order]))
    neurons.u8(np.array([layer_of(cells[r], bands) for r in order]))
    files["neurons.bin"] = neurons.finish(b"CMN1" + u32le(n_neurons))

    decimated = {name: [results[r]["decimated"][name] for r in order] for name in LODS}
    for name, per_neuron in decimated.items():
        counts = np.array([d.count for d in per_neuron])
        if counts.max() >= MAX_NODES:
            raise ValueError(f"{name}: a neuron has {counts.max()} nodes")
        tol, cap, twig = LODS[name]
        log(f"{name} (tolerance {tol} um, max edge {cap} um, twigs under {twig} um dropped): {counts.sum():,} nodes over {n_neurons} neurons "
            f"(min {counts.min()}, median {int(np.median(counts))}, max {counts.max()}); budget {NODE_BUDGET[name]:,}")
        if n_neurons > 1000 and counts.sum() > NODE_BUDGET[name]:
            raise ValueError(f"{name}: {counts.sum():,} nodes is over the {NODE_BUDGET[name]:,} budget; raise its tolerance in LODS")

    # Group neurons into chunks by hi node bytes; lite reuses the grouping.
    groups: list[list[int]] = [[]]
    acc = 0
    for i, d in enumerate(decimated["hi"]):
        size = d.count * BYTES_PER_NODE
        if groups[-1] and acc + size > CHUNK_TARGET_BYTES:
            groups.append([])
            acc = 0
        groups[-1].append(i)
        acc += size

    lods_manifest: dict[str, dict] = {}
    for name, per_neuron in decimated.items():
        starts = np.concatenate([[0], np.cumsum([d.count for d in per_neuron])]).astype(np.int64)
        chunks = []
        for ci, group in enumerate(groups):
            node_start = int(starts[group[0]])
            node_count = int(starts[group[-1] + 1] - node_start)
            parent = np.concatenate([
                np.where(per_neuron[i].parent < 0, NO_PARENT, per_neuron[i].parent + starts[i]).astype(np.uint64)
                for i in group
            ])
            quant = np.concatenate([quantise(per_neuron[i].pos, lo, hi) for i in group])
            radius = np.concatenate([
                np.clip(np.floor(per_neuron[i].radius * 1000 + 0.5), 0, 65535) for i in group
            ])
            path_q = np.concatenate([np.floor(per_neuron[i].dist * 4 + 0.5) for i in group])
            if path_q.max() > 65535:
                raise ValueError(f"{name}: path distance exceeds the u16 quarter-um range")
            comp = np.concatenate([per_neuron[i].comp for i in group])
            ranges = np.array(
                [[i, starts[i], per_neuron[i].count] for i in group], dtype=np.uint32
            ).reshape(-1)

            w = Writer(16)
            w.u32(ranges)
            w.u32(parent)
            w.u16(quant.reshape(-1))
            w.u16(radius)
            w.u16(path_q)
            w.u8(comp)
            rel = f"chunks/{name}/{ci}.bin"
            files[rel] = w.finish(b"CMM1" + u32le(len(group), node_count, node_start))

            deq = dequantise(quant, lo, hi)
            chunks.append({
                "file": rel,
                "nodeStart": node_start,
                "nodeCount": node_count,
                "neuronCount": len(group),
                "bboxUm": {
                    "min": [math.floor(v * 1000) / 1000 for v in deq.min(axis=0)],
                    "max": [math.ceil(v * 1000) / 1000 for v in deq.max(axis=0)],
                },
            })
        lods_manifest[name] = {"nodeCount": int(starts[-1]), "chunks": chunks}
        size = sum(len(v) for k, v in files.items() if k.startswith(f"chunks/{name}/"))
        log(f"{name}: {len(chunks)} chunks, {size:,} bytes")

    synapse_info = None
    if rows is not None:
        files["synapses.bin"], synapse_info = pack_synapses(rows, results, order, lo, hi, log)

    manifest = {
        "version": 1,
        "boundsUm": {"min": lo, "max": hi},
        "neuronCount": n_neurons,
        "cellTypes": cell_types,
        "neurons": "neurons.bin",
        "lods": lods_manifest,
        "synapses": synapse_info,
        "credits": CREDITS,
    }
    files["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    log(f"neurons.bin: {len(files['neurons.bin']):,} bytes; manifest.json: {len(files['manifest.json']):,} bytes")
    return files


def pack_synapses(rows, results, order, lo, hi, log) -> tuple[bytes, dict]:
    index_of = {r: i for i, r in enumerate(order)}
    n_neurons = len(order)
    n_rows = len(rows["id"])
    pre_idx = np.array([index_of[r] for r in rows["pre"]], dtype=np.int64)
    post_idx = np.array([index_of[r] for r in rows["post"]], dtype=np.int64)

    # Path distance to the pre neuron's axon, gathered back from the per-neuron results (which
    # were computed against that neuron's rows in file order).
    pre_dist = np.zeros(n_rows)
    has_axon = np.zeros(n_rows, dtype=bool)
    fallback = np.zeros(n_rows, dtype=bool)
    for r in order:
        res = results[r]
        if res["preDist"] is None:
            continue
        sel = np.flatnonzero(rows["pre"] == r)
        pre_dist[sel] = res["preDist"]
        fallback[sel] = res["fallback"]
        has_axon[sel] = True
    dropped = int((~has_axon).sum())
    # A synapse a cell makes onto itself (an autapse) is real in the table, but in a point-neuron model
    # it is a self-loop that only inflates a cell's own input and its synapse counts. Dropped, and said so.
    autapse = pre_idx == post_idx
    autapses_dropped = int((has_axon & autapse).sum())
    keep = np.flatnonzero(has_axon & ~autapse)
    if len(keep) == 0:
        raise ValueError("no synapse has a presynaptic axon")

    ids, pre_idx, post_idx = rows["id"][keep], pre_idx[keep], post_idx[keep]
    um, size, pre_dist, fallback = rows["um"][keep], rows["size"][keep], pre_dist[keep], fallback[keep]

    outside = int(((um < np.array(lo)) | (um > np.array(hi))).any(axis=1).sum())
    um = np.clip(um, lo, hi)
    pos_q = quantise(um, lo, hi)
    dist_q = np.floor(pre_dist * 4 + 0.5)
    if dist_q.max() > 65535:
        raise ValueError("a presynaptic path distance exceeds the u16 quarter-um range")
    size_q = np.floor(255 * np.log1p(size) / np.log1p(size.max()) + 0.5)

    order_by = np.lexsort((ids, pre_idx))  # by pre neuron, then by synapse id: a fixed order
    pre_sorted = pre_idx[order_by]
    offsets = np.searchsorted(pre_sorted, np.arange(n_neurons + 1), side="left")

    w = Writer(12)
    w.u32(offsets)
    w.u16(pre_sorted)
    w.u16(post_idx[order_by])
    w.u16(pos_q[order_by].reshape(-1))
    w.u16(dist_q[order_by])
    w.u8(size_q[order_by])
    count = len(keep)
    data = w.finish(b"CMS1" + u32le(count, n_neurons))
    log(f"synapses: {n_rows:,} between selected neurons; {dropped:,} dropped (presynaptic neuron has no axon "
        f"nodes); {count:,} packed; {int(fallback.sum()):,} placed on the nearest "
        f"non-axon node (no axon within {AXON_MATCH_UM} um); {autapses_dropped:,} autapses dropped; "
        f"{outside:,} clamped into the bounds; size max {size.max():,.0f}")
    log(f"synapses.bin: {len(data):,} bytes")
    return data, {"file": "synapses.bin", "count": count}


# Hand-written files in public/data that the pack neither writes nor removes.
KEEP = {"README.md"}


def existing_files() -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    for path in sorted(c.OUT.rglob("*")):
        if path.is_file() and path.relative_to(c.OUT).as_posix() not in KEEP:
            found[path.relative_to(c.OUT).as_posix()] = path.read_bytes()
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if public/data differs from a fresh pack")
    args = ap.parse_args()

    t0 = time.time()
    files = pack(log=(lambda *_: None) if args.check else print)
    have = existing_files()

    if args.check:
        stale = sorted(
            [f"changed: {k}" for k in files if k in have and have[k] != files[k]]
            + [f"missing: {k}" for k in files if k not in have]
            + [f"unexpected: {k}" for k in have if k not in files]
        )
        if stale:
            print("public/data is out of date with pipeline/cache:\n  " + "\n  ".join(stale)
                  + "\nrun: .venv/Scripts/python pipeline/pack.py")
            return 1
        print(f"public/data matches a fresh pack ({len(files)} files, {time.time() - t0:.1f}s)")
        return 0

    for k in have:
        if k not in files:
            (c.OUT / k).unlink()
    for k, data in files.items():
        path = c.OUT / k
        path.parent.mkdir(parents=True, exist_ok=True)
        if have.get(k) != data:
            path.write_bytes(data)
    print(f"wrote {len(files)} files to public/data in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
