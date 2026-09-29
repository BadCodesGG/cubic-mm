/**
 * Runs `model.ts` off the main thread for the CPU path (WebGL2 lite mode, or `?sim=cpu`).
 *
 * In:  `init` once, then `step` messages carrying substeps and stimuli.
 * Out: one `stepped` reply per `step`, with a fresh copy of the spike-time ring (transferred, not
 *      copied) and the substeps' spikes and sampled arrivals.
 */

import { SpikingModel, type ModelArrival, type ModelInit, type ModelSpike } from "./model";

export type WorkerIn =
  | { type: "init"; init: ModelInit }
  | { type: "step"; substeps: [t: number, dt: number][]; stimuli: [neuron: number, amount: number][] };

export interface WorkerOut {
  type: "stepped";
  times: Float32Array;
  spikes: ModelSpike[];
  arrivals: ModelArrival[];
  arrivalCount: number;
}

let model: SpikingModel | null = null;

self.onmessage = (e: MessageEvent<WorkerIn>) => {
  const msg = e.data;
  if (msg.type === "init") {
    model = new SpikingModel(msg.init);
    return;
  }
  if (!model) return;
  for (const [n, amount] of msg.stimuli) model.stimulate(n, amount);
  const spikes: ModelSpike[] = [];
  const arrivals: ModelArrival[] = [];
  let arrivalCount = 0;
  for (const [t, dt] of msg.substeps) {
    const r = model.step(t, dt);
    for (const s of r.spikes) spikes.push(s);
    for (const a of r.arrivals) arrivals.push(a);
    arrivalCount += r.arrivalCount;
  }
  const times = model.spikeTimes.slice();
  const out: WorkerOut = { type: "stepped", times, spikes, arrivals, arrivalCount };
  // Typed structurally: the project compiles against the DOM lib, not the worker one.
  const scope = self as unknown as { postMessage(m: WorkerOut, o: { transfer: Transferable[] }): void };
  scope.postMessage(out, { transfer: [times.buffer] });
};
