/**
 * The simulation facade: picks the GPU twin (WebGPU compute) or the CPU worker, wires the synapse
 * table (the dataset's, or a Peters'-rule stand-in when it has none), and turns each frame's
 * spikes and arrivals into `spike`, `arrive` and `mode` events on the bus.
 *
 * Frames longer than 1/60 s are split into equal substeps, so the membrane integration never
 * takes a step much larger than the time constant. Before t = 0 the model runs a silent
 * 2 s warm-up, so the volume already carries pulses and afterglow on the first frame.
 */

import type { StorageBufferAttribute, WebGPURenderer } from "three/webgpu";
import type { Dataset } from "../data";
import type { EventBus } from "../events";
import type { SynapseTable } from "../format";
import { rng } from "../synth";
import { CpuSimulation, type ArraySpikeTimes } from "./cpu";
import { GpuSimulation, type GpuEvents } from "./gpu";
import { STIMULUS, type ModelParams } from "./model";
import { petersSynapses } from "./peters";

export type { ArraySpikeTimes } from "./cpu";

/** Where the renderer reads spike times from: a GPU storage buffer, or an array it uploads. */
export type SpikeTimesSource = { kind: "storage"; attribute: StorageBufferAttribute } | ArraySpikeTimes;

export interface SimulationOptions {
  renderer: WebGPURenderer;
  isWebGPU: boolean;
  dataset: Dataset;
  bus: EventBus;
  seed: number;
  /** Run the CPU worker even on WebGPU (`?sim=cpu`). */
  forceCpu?: boolean;
  params?: Partial<ModelParams>;
}

export interface Simulation {
  mode: "gpu" | "cpu";
  /** Advances the model to simulation time `t`; `dt` is the time since the last call. */
  step(t: number, dt: number): void;
  /** Injects current into a neuron on the next step and watches its cascade. */
  stimulate(neuron: number, amount?: number): void;
  spikeTimesSource: SpikeTimesSource;
  params: ModelParams;
  /** The synapse table the model runs on, and whether it is the Peters'-rule stand-in. */
  synapses: SynapseTable;
  syntheticSynapses: boolean;
  /** Spikes reported since t = 0, how many a stimulus caused, and the recent rate per neuron. */
  stats: { spikes: number; stimulated: number; rateHz: number };
  /** Spikes per neuron since t = 0, as reported back so far. */
  spikeCounts(): Uint32Array;
  /** Post neurons reached so far by pulses from the most recently stimulated neuron, sorted. */
  firstHop(): number[];
  /** Resolves once every result already queued has been reported. */
  settle(): Promise<void>;
  dispose(): void;
}

const MAX_SUBSTEP = 1 / 60;
const WARMUP_S = 2;
const RATE_WINDOW_S = 2;

interface Backend {
  params: ModelParams;
  substep(t: number, dt: number): void;
  endFrame(): void;
  stimulate(neuron: number, amount: number): void;
  drain(): GpuEvents[];
  settle(): Promise<void>;
  dispose(): void;
}

export function createSimulation(opts: SimulationOptions): Simulation {
  const { dataset, bus, seed } = opts;
  const neuronCount = dataset.neurons.count;
  const syntheticSynapses = dataset.synapses === null;
  const synapses = dataset.synapses ?? petersSynapses(dataset, rng(seed ^ 0x5eed));
  const init = { neuronCount, inhibitory: dataset.neurons.inhibitory, synapses, seed, params: opts.params };

  const gpu = opts.isWebGPU && !opts.forceCpu;
  let backend: Backend;
  let spikeTimesSource: SpikeTimesSource;
  if (gpu) {
    const g = new GpuSimulation(opts.renderer, init);
    backend = g;
    spikeTimesSource = { kind: "storage", attribute: g.spikeTimes };
  } else {
    const c = new CpuSimulation(init);
    backend = c;
    spikeTimesSource = c.source;
  }

  // Silent warm-up from -WARMUP_S to 0, reported back as one batch that is discarded.
  const warmSteps = Math.round(WARMUP_S / MAX_SUBSTEP);
  for (let i = 1; i <= warmSteps; i++) backend.substep(-WARMUP_S + i * MAX_SUBSTEP, MAX_SUBSTEP);
  backend.endFrame();
  let discard = 1;

  const counts = new Uint32Array(neuronCount);
  const reached = new Set<number>();
  let watched = -1;
  const recent: [time: number, count: number][] = [];
  let lastTime = 0;
  let announced = false;
  const stats = { spikes: 0, stimulated: 0, rateHz: 0 };

  const report = (batch: GpuEvents) => {
    let latest = lastTime;
    for (const s of batch.spikes) {
      counts[s.neuron]++;
      if (s.stimulated) stats.stimulated++;
      if (s.time > latest) latest = s.time;
      bus.emit("spike", s);
    }
    for (const a of batch.arrivals) {
      const pre = synapses.pre[a.synapse];
      const post = synapses.post[a.synapse];
      if (pre === watched) reached.add(post);
      bus.emit("arrive", { pre, post, synapse: a.synapse, time: a.time });
    }
    stats.spikes += batch.spikeCount;
    lastTime = latest;
    recent.push([lastTime, batch.spikeCount]);
    while (recent.length && recent[0][0] < lastTime - RATE_WINDOW_S) recent.shift();
    const inWindow = recent.reduce((sum, [, c]) => sum + c, 0);
    stats.rateHz = inWindow / Math.min(RATE_WINDOW_S, Math.max(lastTime, 1e-3)) / Math.max(1, neuronCount);
  };

  const sim: Simulation = {
    mode: gpu ? "gpu" : "cpu",
    params: backend.params,
    synapses,
    syntheticSynapses,
    spikeTimesSource,
    stats,
    step(t, dt) {
      if (!announced) {
        announced = true;
        bus.emit("mode", { gpu, neuronCount, synapseCount: synapses.count, syntheticSynapses });
      }
      if (dt > 0) {
        const n = Math.max(1, Math.ceil(dt / MAX_SUBSTEP - 1e-6));
        for (let i = 1; i <= n; i++) backend.substep(t - dt + (dt * i) / n, dt / n);
        backend.endFrame();
      }
      for (const batch of backend.drain()) {
        if (discard > 0) {
          discard--;
          continue;
        }
        report(batch);
      }
    },
    stimulate(neuron, amount = STIMULUS) {
      if (neuron !== watched) reached.clear();
      watched = neuron;
      backend.stimulate(neuron, amount);
    },
    spikeCounts: () => counts.slice(),
    firstHop: () => [...reached].sort((a, b) => a - b),
    async settle() {
      await backend.settle();
      sim.step(lastTime, 0);
    },
    dispose() {
      backend.dispose();
    },
  };
  return sim;
}
