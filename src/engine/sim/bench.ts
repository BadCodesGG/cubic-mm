/**
 * Measures what the simulation costs per 60 Hz frame at a given size, on this device, for
 * `scripts/parity.mjs`. Builds a random table (sorted by pre, delays across a 1.3 mm axon), then:
 *   GPU: one substep plus the event readback per frame, back to back, until every readback has
 *        landed; wall time / frames. Submission overhead included, so it is an upper bound.
 *   CPU: `model.step` on the main thread, the same code the worker runs.
 */

import type { WebGPURenderer } from "three/webgpu";
import type { SynapseTable } from "../format";
import { rng } from "../synth";
import { GpuSimulation } from "./gpu";
import { SpikingModel, type ModelInit } from "./model";

export interface BenchResult {
  neurons: number;
  synapses: number;
  gpuMsPerFrame: number | null;
  cpuMsPerFrame: number;
  spikesPerSecond: number;
}

function randomTable(neurons: number, synapses: number, seed: number): SynapseTable {
  const r = rng(seed);
  const pre = new Uint16Array(synapses);
  const preOffsets = new Uint32Array(neurons + 1);
  for (let s = 0; s < synapses; s++) pre[s] = Math.floor((s / synapses) * neurons);
  for (let s = 0, n = 0; n <= neurons; n++) {
    while (s < synapses && pre[s] < n) s++;
    preOffsets[n] = s;
  }
  const post = Uint16Array.from(pre, (p) => (p + 1 + Math.floor(r() * (neurons - 1))) % neurons);
  const preDistQ = Uint16Array.from(pre, () => Math.floor(r() * 1300 * 4));
  const size = Uint8Array.from(pre, () => Math.floor(r() * 256));
  return { count: synapses, neuronCount: neurons, preOffsets, pre, post, pos: new Uint16Array(synapses * 3), preDistQ, size };
}

export async function benchSimulation(
  renderer: WebGPURenderer | null,
  neurons: number,
  synapses: number,
  frames = 300,
): Promise<BenchResult> {
  const inhibitory = Uint8Array.from({ length: neurons }, (_, i) => (i % 5 === 0 ? 1 : 0));
  const init: ModelInit = { neuronCount: neurons, inhibitory, synapses: randomTable(neurons, synapses, 3), seed: 5 };
  const dt = 1 / 60;

  let gpuMsPerFrame: number | null = null;
  if (renderer) {
    const gpu = new GpuSimulation(renderer, init);
    for (let i = 1; i <= 10; i++) {
      gpu.substep(i * dt, dt);
      gpu.endFrame();
    }
    await gpu.settle();
    const t0 = performance.now();
    for (let i = 11; i <= 10 + frames; i++) {
      gpu.substep(i * dt, dt);
      gpu.endFrame();
    }
    await gpu.settle();
    gpuMsPerFrame = (performance.now() - t0) / frames;
    gpu.dispose();
  }

  const cpu = new SpikingModel(init);
  let spikes = 0;
  for (let i = 1; i <= 60; i++) spikes += cpu.step(i * dt, dt).spikes.length;
  const t0 = performance.now();
  for (let i = 61; i <= 60 + frames; i++) spikes += cpu.step(i * dt, dt).spikes.length;
  const cpuMsPerFrame = (performance.now() - t0) / frames;
  return { neurons, synapses, gpuMsPerFrame, cpuMsPerFrame, spikesPerSecond: spikes / ((60 + frames) * dt) };
}
