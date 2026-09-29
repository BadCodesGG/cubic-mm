/**
 * The CPU path's main-thread half: owns the worker running `model.ts`, batches a frame's
 * substeps into one message, and keeps the latest spike-time ring for the renderer to upload.
 *
 * At most one message is in flight. If the worker is still busy when a frame ends, that frame's
 * substeps wait and go with the next one, so a slow device falls behind smoothly instead of
 * queueing unbounded work.
 */

import { DEFAULT_PARAMS, EMPTY_SPIKE, SPIKE_SLOTS, type ModelInit, type ModelParams } from "./model";
import type { GpuEvents } from "./gpu";
import type { WorkerIn, WorkerOut } from "./worker";

export interface ArraySpikeTimes {
  kind: "array";
  /** 8 spike times per neuron, replaced (not mutated) each time the worker replies. */
  times: Float32Array;
  /** Bumped on every replacement, so a consumer uploads only when it changed. */
  version: number;
}

export class CpuSimulation {
  readonly params: ModelParams;
  readonly source: ArraySpikeTimes;

  private readonly worker: Worker;
  private substeps: [number, number][] = [];
  private stimuli: [number, number][] = [];
  private busy = false;
  private readonly ready: GpuEvents[] = [];

  constructor(init: ModelInit) {
    this.params = { ...DEFAULT_PARAMS, ...init.params };
    this.source = { kind: "array", times: new Float32Array(init.neuronCount * SPIKE_SLOTS).fill(EMPTY_SPIKE), version: 0 };
    this.worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (e: MessageEvent<WorkerOut>) => this.receive(e.data);
    this.worker.onerror = (e) => console.error("Simulation worker failed:", e.message);
    this.post({ type: "init", init });
  }

  stimulate(neuron: number, amount: number): void {
    this.stimuli.push([neuron, amount]);
  }

  substep(t: number, dt: number): void {
    this.substeps.push([t, dt]);
  }

  endFrame(): void {
    if (this.busy || this.substeps.length === 0) return;
    this.busy = true;
    this.post({ type: "step", substeps: this.substeps, stimuli: this.stimuli });
    this.substeps = [];
    this.stimuli = [];
  }

  drain(): GpuEvents[] {
    return this.ready.splice(0);
  }

  async settle(): Promise<void> {
    while (this.busy || this.substeps.length > 0) {
      this.endFrame();
      await new Promise((r) => setTimeout(r, 5));
    }
  }

  dispose(): void {
    this.worker.terminate();
  }

  private post(msg: WorkerIn): void {
    this.worker.postMessage(msg);
  }

  private receive(msg: WorkerOut): void {
    this.busy = false;
    this.source.times = msg.times;
    this.source.version++;
    this.ready.push({
      spikes: msg.spikes,
      arrivals: msg.arrivals.map((a) => ({ synapse: a.synapse, time: a.time })),
      spikeCount: msg.spikes.length,
      arrivalCount: msg.arrivalCount,
    });
  }
}
