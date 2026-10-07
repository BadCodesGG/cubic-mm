# One Cubic Millimetre

**Walk through a real cubic millimetre of mouse brain.**
1,711 real neurons, 156,882 real synapses, and a live spiking simulation you can set off, running on WebGPU in the browser.

**Live: [cubic.badcodes.dev](https://cubic.badcodes.dev)**

https://github.com/user-attachments/assets/73ad80d3-ebe8-4492-87b4-816e7e8b3f73

The data is the [MICrONS](https://www.microns-explorer.org/cortical-mm3) minnie65 volume: a cubic millimetre of mouse visual cortex, the only piece of brain ever mapped down to every synapse. The proofread skeletons of 1,711 of its neurons are rendered as glowing ribbons at true scale, and a leaky integrate-and-fire model runs over 156,882 of their real synapses. Click a cell and stimulate it, watch the signal travel down its actual axon, see which cells it fires, and ride the spike into the next one. Every spike is a synthesised sound placed in 3D.

The wiring is measured. The dynamics on it are a toy, and the site's [about page](https://cubic.badcodes.dev/about) says exactly which is which.

## What you can do

| | |
|---|---|
| **Fly** | WASD and the mouse, `Shift` to go faster. On a phone, two thumbsticks. |
| **Select a cell** | Click a soma. The panel shows its type, layer, depth below the pia, cable length, synapse counts and MICrONS root id. |
| **Stimulate it** | `Space`. The spike leaves the soma, runs down the real axon at 0.5 m/s slowed 1000x, and lights up the cells it reaches. |
| **Watch the cascade** | Lines draw from each driver to each cell it fired; the panel counts hops and shows an 8 s timeline. |
| **Ride the spike** | `R`. The camera drops onto the pulse, follows it to a synapse, and crosses into the next cell. |
| **Search** | `/`. Root id, type code (`4P`, `BC`, `MC`), `inhibitory`, `layer 5`, or a chip like "a Martinotti cell". |
| **Share the view** | The address bar always holds the camera and selection. Copy link, send it, they see what you see. |
| **Slow time** | `,` for 0.1x, `.` for normal, Pause in the HUD. The camera stays live. |
| **Screenshot** | `P` saves a 3200 px PNG with the HUD hidden, named after the selected cell. |
| **Home** | `H` flies back to the start. The minimap shows where you are in the volume. |

A 20 s intro plays on the first visit and can be replayed from the About panel. Sound is off until you turn it on.

## How it works

```
pipeline/            Python + numpy. Fetches the public MICrONS files, streams the 20 GB synapse
                     table once, decimates the skeletons to two detail tiers, packs 24 MB of binary.
public/data/         The packed dataset, committed, so the site runs without Python.
src/engine/          The experience, plain TypeScript, no framework in the hot path.
  format.ts          The binary contract the pipeline writes and the browser reads.
  sim/               Leaky integrate-and-fire: TSL compute kernels on WebGPU, and the identical
                     model in a Web Worker for the WebGL2 fallback.
  scene/             Instanced ribbons, somas, pulse fronts, cascade lines, bloom, haze.
  camera/            Fly controls, the ride-a-spike camera, eased flights, the intro tour.
  audio/             Web Audio synthesis: one HRTF-panned grain per spike, no samples.
src/components/      The React HUD, panels, search, minimap and the about page.
scripts/             Smoke test, GPU/CPU parity check, reproducible screenshots.
```

**Rendering.** three.js r186 on the `WebGPURenderer`, materials written in TSL so one shader graph serves both backends. Every skeleton edge is an instanced, view-aligned ribbon; the spike glow is a travelling front computed from each node's path distance to the soma, so a small buffer of spike times animates over two million segments with no per-frame CPU work. Post-processing is bloom, grain and vignette through the `RenderPipeline`.

**Simulation.** Each frame a synapse kernel (one thread per synapse) checks the presynaptic cell's last eight spike times for a pulse landing in this step and accumulates the weight into the target with an atomic add; a neuron kernel integrates, thresholds, and writes spike times back into the buffer the ribbons read. A visitor's stimulus is delivered with extra gain that decays hop by hop, so a cascade is visible; the background dynamics are untouched. `scripts/parity.mjs` proves the GPU and CPU paths agree.

**Quality tiers.** The app starts at the highest detail the device is likely to hold and steps down (fewer curve pieces, then the coarser skeleton tier) if the measured frame time says so, without mistaking a 30 Hz display for a slow GPU. Browsers without WebGPU get the WebGL2 renderer and the worker simulation.

### Things to know before changing the code

- Never use `ShaderMaterial`, `onBeforeCompile` or `EffectComposer`: they do not run on the WebGPU backend. Materials are TSL node materials, and post-processing is a `RenderPipeline` with TSL display nodes from `three/examples/jsm/tsl/display/`.
- Gate compute on `renderer.backend.isWebGPUBackend`. The WebGL2 fallback compiles TSL to GLSL but has no real compute, so lite mode runs the same simulation in a Web Worker (`src/engine/sim/model.ts`).
- Do not construct `THREE.Clock`. three r186 deprecates it with a console warning, and the smoke test fails on any `THREE.` warning.
- `src/engine/format.ts` is the binary data contract and `pipeline/pack.py` mirrors it byte for byte. `format.test.ts` round-trips the TypeScript encoders and `data-integrity.test.ts` checks the committed files: change one side, change the other and run both.
- Data is in µm with y as depth (the pia at low y). Positions are u16 quantised against `manifest.boundsUm`, path distance is quarter-µm u16 and radius is nm u16.
- The MICrONS credit is required by its licence: the manifest carries the citation and `/about` must show it.
- Any route that sets metadata goes through `pageMetadata()` in `src/lib/site.ts`. A route that declares its own `openGraph` or `twitter` replaces the layout's wholesale, so its card loses the picture; a route that sets only `title` unfurls with the home page's title and URL. The production origin must be a literal domain, never `VERCEL_URL` or localhost, because a relative or deployment URL makes Slack and Facebook drop the image.
- Type-check with `npm run typecheck`, not bare `tsc`: it generates the route types first, and bare `tsc` fails on a fresh checkout.

## Running it

Node 22 or 24.

```bash
git clone https://github.com/BadCodesGG/cubic-mm.git
cd cubic-mm
npm install
npm run dev            # http://localhost:3000
```

Checks:

```bash
npm run typecheck      # generates the Next route types, then tsc --noEmit
npm run lint           # ESLint
npm run test           # Vitest: data format, data integrity, the CPU model, camera and HUD logic
npm run build          # production build
npm run test:smoke     # boots the build, drives it in Chromium with WebGPU on and off, runs parity
```

The smoke test needs a build first and Playwright's Chromium (`npx playwright install chromium`). On a machine without a GPU, `npm run test:smoke -- --allow-software`. The social card is the static `public/og.jpg`; `NEXT_PUBLIC_SITE_URL` overrides its origin (default `https://cubic.badcodes.dev`).

### Regenerating the data

`public/data/` is committed. To rebuild it from the public MICrONS files (Python 3 with numpy in a venv at `.venv`):

```bash
.venv/Scripts/python pipeline/fetch.py --count all   # skeletons and the cell table into pipeline/cache/
.venv/Scripts/python pipeline/synapses.py            # streams the 337 M row synapse table once, resumable
.venv/Scripts/python pipeline/pack.py                # writes public/data/; --check fails if it is stale
```

[`pipeline/README.md`](pipeline/README.md) documents the decimation, the synapse filtering, and the binary layout shared with `src/engine/format.ts`.

## Data and licence

The neuron skeletons, cell types and synapses come from the MICrONS Consortium and are licensed **CC BY 4.0**.

> The MICrONS Consortium. Functional connectomics spanning multiple areas of mouse visual cortex. *Nature* 640, 435-447 (2025). https://doi.org/10.1038/s41586-025-08790-w

The code is MIT, see [LICENSE](LICENSE).

Files that are not covered by the MIT licence, and the terms for the BadCodes name and logo, are listed in [NOTICE](NOTICE).

Built by [BadCodes](https://badcodes.dev).
