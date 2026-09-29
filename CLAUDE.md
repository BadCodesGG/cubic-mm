# CLAUDE.md

Guidance for Claude Code in this repository.

@AGENTS.md

## What this is

"One Cubic Millimetre": a walkable, first-person experience inside the MICrONS minnie65 cubic
millimetre of mouse visual cortex. Real proofread neuron skeletons (CC BY 4.0) are rendered as
glowing ribbons; a spiking simulation runs over their real synaptic wiring; the visitor can click a
cell, watch the signal travel down its axon, and ride a spike. WebGPU first, with a lite WebGL2 mode.

## Commands

```bash
npm run dev            # Dev server at http://localhost:3000
npm run build          # Production build (the real gate; dev mode hides prerender problems)
npm run lint           # ESLint (Next core-web-vitals + TypeScript)
npm run test           # Vitest: format round-trips, data integrity of public/data, the CPU sim
npm run test:smoke     # Boots the built site, drives it in Chromium with WebGPU on and off
npm run data:check     # Re-packs from pipeline/cache and fails if public/data is stale
.venv/Scripts/python pipeline/fetch.py --count 200   # Download skeletons + cell table into pipeline/cache
.venv/Scripts/python pipeline/pack.py                # Write public/data from the cache
```

Type-check with `npx tsc --noEmit`. Python is only ever the venv at `.venv/Scripts/python`.

## Architecture

- Next.js 16 App Router shell; React owns the HUD, loader and overlays only.
- The scene is vanilla three.js r186 `WebGPURenderer` + TSL in `src/engine/`, no React Three Fiber.
  Engine modules are plain TypeScript and unit-testable without a GPU.
- `src/engine/format.ts` is the binary data contract. `pipeline/pack.py` mirrors it byte for byte;
  `format.test.ts` round-trips the TS encoders, `data-integrity.test.ts` checks the committed files.
  Change one side, change the other, and run both tests.
- `src/engine/data.ts` loads a LOD ("hi" or "lite") into flat arrays with global node indices.

## Things that bite

- **Never `ShaderMaterial`, `onBeforeCompile` or `EffectComposer`.** They do not run on the WebGPU
  backend. Materials are TSL node materials; post-processing is `RenderPipeline` with TSL display
  nodes from `three/examples/jsm/tsl/display/`.
- **Gate compute on `renderer.backend.isWebGPUBackend`.** The WebGL2 fallback compiles TSL to GLSL
  but has no real compute; lite mode runs the same simulation in a Web Worker (`sim/model.ts`).
- **Do not construct `THREE.Clock`.** three 0.186 deprecates it with a console warning, and the
  smoke test fails on any `THREE.` warning.
- **`PCFSoftShadowMap` is gone in r186.** There are no shadows in this scene anyway.
- **Data is coordinates in µm; y is depth (pia at low y).** Positions are quantised u16 against
  `manifest.boundsUm`; path distance is quarter-µm u16; radius is nm u16.
- **Credits are not optional.** The MICrONS data is CC BY 4.0; the manifest carries the citation
  and `/about` must show it.
