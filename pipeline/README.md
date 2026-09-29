# Data pipeline

Turns the public MICrONS proofread skeletons into the binary files the site loads from
`public/data/` (layout: `src/engine/format.ts`). Python is the repo venv only:
`.venv/Scripts/python` (3.14, numpy).

```bash
.venv/Scripts/python pipeline/fetch.py --count 200   # ~10 s, downloads ~75 MB into pipeline/cache/
.venv/Scripts/python pipeline/pack.py                # ~30 s, writes public/data/
.venv/Scripts/python pipeline/pack.py --check        # ~15 s, exits 1 if public/data/ is stale
```

`fetch.py` lists the anonymous `microns-static-links` bucket, downloads the `aibs_cell_info` table,
joins them on root id (unmatched ids are skipped), and keeps the `--count` somas nearest the
centroid of all matched proofread somas (`--count all` keeps every one). Files already in
`pipeline/cache/` with the listed size are not downloaded again. `pipeline/cache/` is gitignored.

`pack.py` reorders each skeleton soma-first with parents before children, computes path length at
full resolution, then decimates per LOD (Ramer-Douglas-Peucker per unbranched run; branch points,
tips, the soma and compartment changes are kept; no edge over 12 um). `hi` tolerance is 0.75 um,
`lite` is 3 um; both are constants at the top of `pack.py`. Neurons are ordered along a Morton curve
of their somas and cut into chunks of about 1.5 MB of `hi` data. The output is a pure function of
`pipeline/cache/`, so `--check` can prove the committed files came from it.

Layers: excitatory from `cell_type` (23P 2, 4P 4, 5P-* 5, 6P-* 6); everything else from soma depth
(y, pia at low y) against y bands computed from all matched excitatory cells.

Data: MICrONS Consortium, CC BY 4.0, https://www.microns-explorer.org/cortical-mm3.
