# Data pipeline

Turns the public MICrONS proofread skeletons into the binary files the site loads from
`public/data/` (layout: `src/engine/format.ts`). Python is the repo venv only:
`.venv/Scripts/python` (3.14, numpy).

```bash
.venv/Scripts/python pipeline/fetch.py --count all   # ~1 min, 711 MB of skeletons + cell table into pipeline/cache/
.venv/Scripts/python pipeline/synapses.py            # ~10 min, streams 20 GB, keeps ~181k synapses (resumable)
.venv/Scripts/python pipeline/pack.py                # ~30 s, writes public/data/ (the committed set is the full one)
.venv/Scripts/python pipeline/pack.py --check        # ~20 s, exits 1 if public/data/ is stale
```

`--count 200` on `fetch.py` selects only the 200 somas nearest the centroid, for a small set (~10 s,
75 MB); `synapses.py` and `pack.py` then follow the selection. `synapses.py` refuses a stale file
for a different selection, and `pack.py` writes no synapses if it has not run.

`fetch.py` lists the anonymous `microns-static-links` bucket, downloads the `aibs_cell_info` table,
joins them on root id (unmatched ids are skipped: 1,711 of the 1,723 proofread skeletons match), and
keeps the `--count` somas nearest the centroid of all matched proofread somas (`all` keeps every
one). Files already in `pipeline/cache/` with the listed size are not downloaded again.
`pipeline/cache/` is gitignored.

`synapses.py` streams `synapses_pni_2_v1_filtered_view.csv.gz` (337 M rows) in 64 MiB ranged reads,
inflates it, and keeps rows whose pre and post root ids are both selected, into
`cache/synapses_selected.csv` (voxel coordinates untouched). A deflate stream cannot be entered
mid-way, so resuming replays the already-done blocks (download and inflate, no filtering), truncates
the csv to the checkpointed size in `cache/synapses_selected.json`, and continues: no row is written
twice.

`pack.py` reorders each skeleton soma-first with parents before children, computes path length at
full resolution, then decimates per LOD (Ramer-Douglas-Peucker per unbranched run; branch points,
tips, the soma and compartment changes are kept). `LODS` at the top of `pack.py` holds, per LOD,
the tolerance, the longest edge and the shortest terminal twig kept; the whole set must fit the
`NODE_BUDGET` (hi 1.2 M, lite 350 k nodes) or the pack fails. Cable totals 16.5 m over 1,711 neurons,
so the budget alone forces about 14 um per node: hi is 2 um / 28 um edges, lite is 8 um / 150 um
edges with terminal twigs under 50 um dropped. Neurons are ordered along a Morton curve of their
somas and cut into chunks of about 1.5 MB of `hi` data. Skeletons are processed in worker
processes but gathered in a fixed order, so the output is a pure function of `pipeline/cache/`
and `--check` can prove the committed files came from it.

Synapses: `pre`/`post` are neuron indices, `pos` is the centre point in um (voxels are 4x4x40 nm),
`preDistQ` is the path distance of the nearest full-resolution axon node of the presynaptic neuron
(nearest node of any compartment when none is within 5 um), `size` is
`round(255 * log1p(size) / log1p(maxSize))`. Synapses whose presynaptic neuron has no axon nodes are
dropped. Autapses (pre == post) are kept.

Layers: excitatory from `cell_type` (23P 2, 4P 4, 5P-* 5, 6P-* 6); everything else from soma depth
(y, pia at low y) against y bands computed from all matched excitatory cells.

Data: MICrONS Consortium, CC BY 4.0, https://www.microns-explorer.org/cortical-mm3.
