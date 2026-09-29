# One Cubic Millimetre

A walkable, first-person experience inside the MICrONS minnie65 cubic millimetre of mouse visual
cortex: 1,711 real proofread neurons drawn as glowing ribbons, with a spiking simulation running over
their 156,882 real synapses. Click a cell, watch the signal travel down its axon, and ride a spike.

![Ribbons of neurons in the volume, one axon carrying a bright pulse](public/hero-still.jpg)

The wiring is measured; the dynamics on it are a toy (leaky integrate-and-fire point neurons, Poisson
background drive). The site's `/about` page says which is which.

WebGPU first (three.js r186 `WebGPURenderer`, the simulation as TSL compute), with a lite WebGL2 mode
that runs the same model in a Web Worker. Built with Next.js 16 and Claude Code.

## Commands

```bash
npm run dev          # Dev server at http://localhost:3000
npm run build        # Production build (the real gate; dev mode hides prerender problems)
npm run lint         # ESLint
npm run test         # Vitest: format round-trips, data integrity of public/data, the CPU sim
npm run test:smoke   # Boots the built site on port 3132 and drives it in Chromium with WebGPU on and off
npm run data:check   # Re-packs from pipeline/cache and fails if public/data is stale
```

Type-check with `npx tsc --noEmit`.

`npm run test:smoke` needs a build first and Playwright's Chromium (`npx playwright install chromium`).
It checks the GPU simulation, a stimulate-and-ride interaction, the WebGL2 fallback, `/about`, the social
image, and finishes with `scripts/parity.mjs` (GPU against CPU spike statistics). On a machine without a
GPU pass `--allow-software`: `npm run test:smoke -- --allow-software`. `NEXT_PUBLIC_SITE_URL` sets the
origin used in social cards.

## Data pipeline

`public/data/` is committed, so the site runs without Python. To regenerate it, use the repo venv
(`.venv/Scripts/python`, Python with numpy):

```bash
.venv/Scripts/python pipeline/fetch.py --count all   # skeletons and the cell table into pipeline/cache/
.venv/Scripts/python pipeline/synapses.py            # streams the 337 M row synapse table once
.venv/Scripts/python pipeline/pack.py                # writes public/data/
```

`pipeline/README.md` has the details: decimation (2.0 µm / 28 µm for the full detail, 8 µm / 150 µm and
short twigs pruned for lite), the synapse filtering, and the binary format shared with
`src/engine/format.ts`.

## Licence and citation

Neuron skeletons, cell types and synapses: MICrONS Consortium, CC BY 4.0,
<https://www.microns-explorer.org/cortical-mm3>.

> The MICrONS Consortium. Functional connectomics spanning multiple areas of mouse visual cortex.
> Nature 640, 435-447 (2025). https://doi.org/10.1038/s41586-025-08790-w
