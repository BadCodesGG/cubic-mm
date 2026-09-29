/**
 * The spiking model: leaky integrate-and-fire neurons over the synapse table, in plain
 * TypeScript with no three.js, so it runs in a Web Worker (lite mode, `?sim=cpu`) and in unit
 * tests. `gpu.ts` is the same algorithm as two TSL compute kernels; keep the two in step.
 *
 * Time is display time in seconds. Membrane dynamics run at real speed (tau 20 ms), but the
 * axonal travel delay is stretched by `slowMo`, so a pulse crossing 800 µm of axon takes 1.6 s on
 * screen instead of 1.6 ms, which is what makes it visible travelling.
 *
 * Each step has two passes:
 *   1. Synapses: for every synapse, for each of its presynaptic neuron's last 8 spike times,
 *      a pulse arriving inside (t - dt, t] adds the synapse's weight to the post neuron's input.
 *   2. Neurons: Poisson background, integrate with exact exponential decay, threshold, reset,
 *      refractory period (during which v is held at vReset and input is discarded).
 *
 * The driven cascade. With the delay stretched 1000x and tau left at 20 ms, pulses from different
 * cells land seconds apart and never sum, so one ordinary spike (weight 0.08 to 0.16) cannot fire
 * anything. A visitor's stimulus therefore carries a drive generation: the stimulated spike is
 * generation 0, and a spike fired in the same step an excitatory generation-g pulse landed is
 * generation g + 1. A generation-g pulse delivers `weight * drivenGain[g]` (12, 4, 2, 1, then 1), so
 * hop 1 fires reliably, hop 2 when the wiring is dense, and the cascade dies out by itself.
 * Every other spike is generation NO_GENERATION and delivers its weight x1 exactly, so the
 * background dynamics are the same as without the mechanism.
 */

import { PATH_DIST_UNIT_UM, type SynapseTable } from "../format";
import { rng } from "../synth";

export const SPIKE_SLOTS = 8;
/** An empty spike slot: long enough ago that nothing it drives can still be happening. */
export const EMPTY_SPIKE = -1e9;
/** Most non-watched arrivals one step reports; the audio layer does not need every one. */
export const ARRIVAL_SAMPLE = 64;
/**
 * Current `stimulate` injects by default. A stimulated cell fires on its next non-refractory step
 * whatever its potential: a hub cell under heavy inhibition sat near -1 and let a +2 kick fall short,
 * so the visitor's click did nothing. The amount still reaches the input, and flags the spike.
 */
export const STIMULUS = 2;
/** The generation of an ordinary spike, one no stimulus caused; also an empty slot's. */
export const NO_GENERATION = 255;

/** The generation of a spike fired while the lowest driven generation arriving was `drivenIn`. */
export function nextGeneration(stimulated: boolean, drivenIn: number): number {
  return stimulated ? 0 : Math.min(drivenIn + 1, NO_GENERATION);
}

/** The gain a pulse of each generation (0..255) is delivered at: `drivenGain[g]`, else 1. */
export function drivenGainTable(p: Pick<ModelParams, "drivenGain">): Float32Array {
  const out = new Float32Array(NO_GENERATION + 1).fill(1);
  p.drivenGain.forEach((g, i) => {
    if (i < NO_GENERATION) out[i] = g;
  });
  return out;
}

export interface ModelParams {
  tauMs: number;
  vRest: number;
  vThreshold: number;
  vReset: number;
  refractoryMs: number;
  /** Poisson background drive per neuron, events per second, so the volume is never dead. */
  backgroundRateHz: number;
  /**
   * Input each background event adds. Above 1.0 (vThreshold - vRest) an event fires a resting
   * cell; recent inhibition can still veto it. Exactly 1.0 let the float residue of inhibition
   * from seconds earlier veto most events. At 1.1 and 0.5 Hz, the real 1,711-cell wiring fires
   * about 0.33 Hz per neuron: the rest is vetoed by inhibition from the last few tens of ms.
   */
  backgroundWeight: number;
  excWeight: number;
  inhWeight: number;
  /** Weight scales by `1 + sizeGain * size / 255`. */
  sizeGain: number;
  /** Axonal conduction velocity, m/s. */
  conductionMps: number;
  /** Stretches the travel delay only. */
  slowMo: number;
  synDelayMs: number;
  /** Weight multiplier for a pulse of drive generation 0, 1, 2...; 1 past the end and for ordinary spikes. */
  drivenGain: number[];
}

export const DEFAULT_PARAMS: ModelParams = {
  tauMs: 20,
  vRest: 0,
  vThreshold: 1,
  vReset: 0,
  refractoryMs: 4,
  backgroundRateHz: 0.5,
  backgroundWeight: 1.1,
  excWeight: 0.08,
  inhWeight: -0.12,
  sizeGain: 1.0,
  conductionMps: 0.5,
  slowMo: 1000,
  synDelayMs: 1,
  drivenGain: [12, 4, 2, 1],
};

/** The parts of the synapse table the model needs (positions are only for drawing). */
export type ModelSynapses = Pick<SynapseTable, "count" | "preOffsets" | "pre" | "post" | "preDistQ" | "size">;

export interface ModelInit {
  neuronCount: number;
  /** 1 if inhibitory: sets the sign of every synapse the neuron makes. */
  inhibitory: Uint8Array;
  synapses: ModelSynapses;
  seed: number;
  params?: Partial<ModelParams>;
}

export interface ModelSpike {
  neuron: number;
  time: number;
  stimulated: boolean;
  /** Drive generation: 0 for the stimulated spike, g + 1 when a generation-g pulse fired it, NO_GENERATION otherwise. */
  generation: number;
}

export interface ModelArrival {
  pre: number;
  post: number;
  synapse: number;
  /** Exact arrival time, inside the step's window. */
  time: number;
}

export interface StepResult {
  spikes: ModelSpike[];
  /** Every arrival from the watched neuron or from a driven spike, plus up to ARRIVAL_SAMPLE others. */
  arrivals: ModelArrival[];
  /** Total arrivals this step, including the ones not sampled. */
  arrivalCount: number;
}

/** Travel delay to each synapse in seconds: along the axon at conduction speed, slowed, plus the synaptic delay. */
export function synapseDelays(synapses: ModelSynapses, p: ModelParams): Float32Array {
  const perUm = (1e-6 / p.conductionMps) * p.slowMo;
  const out = new Float32Array(synapses.count);
  for (let s = 0; s < synapses.count; s++) {
    out[s] = synapses.preDistQ[s] * PATH_DIST_UNIT_UM * perUm + p.synDelayMs / 1000;
  }
  return out;
}

/** Signed weight of each synapse: sign from the presynaptic cell, magnitude scaled by cleft size. */
export function synapseWeights(synapses: ModelSynapses, inhibitory: Uint8Array, p: ModelParams): Float32Array {
  const out = new Float32Array(synapses.count);
  for (let s = 0; s < synapses.count; s++) {
    const base = inhibitory[synapses.pre[s]] ? p.inhWeight : p.excWeight;
    out[s] = base * (1 + (p.sizeGain * synapses.size[s]) / 255);
  }
  return out;
}

export class SpikingModel {
  readonly params: ModelParams;
  readonly neuronCount: number;
  readonly v: Float32Array;
  readonly refractoryUntil: Float64Array;
  /** Last SPIKE_SLOTS spike times per neuron, seconds; neuron n owns [n * 8, n * 8 + 8). */
  readonly spikeTimes: Float32Array;
  /** Drive generation of each spike slot, parallel to `spikeTimes`; NO_GENERATION when empty or ordinary. */
  readonly spikeGen: Uint8Array;
  /** Next slot each neuron writes. */
  readonly spikeHead: Uint8Array;
  readonly input: Float32Array;
  /** Arrivals whose presynaptic cell is this one are always reported (the visitor's cascade). */
  watched = -1;

  private readonly syn: ModelSynapses;
  private readonly delay: Float32Array;
  private readonly weight: Float32Array;
  private readonly gain: Float32Array;
  /** Lowest generation of any excitatory driven pulse landing on each neuron this step. */
  private readonly drivenIn: Uint8Array;
  /** Longest delay out of each neuron: slots older than this cannot deliver anything. */
  private readonly maxDelay: Float32Array;
  private readonly minDelay: Float32Array;
  private readonly stimulated: Uint8Array;
  /** What each pending stimulus injected, so it can be re-queued past a refractory window. */
  private readonly pendingStimulus: Float32Array;
  private readonly random: () => number;

  constructor(init: ModelInit) {
    this.params = { ...DEFAULT_PARAMS, ...init.params };
    const n = init.neuronCount;
    this.neuronCount = n;
    this.syn = init.synapses;
    this.v = new Float32Array(n).fill(this.params.vRest);
    this.refractoryUntil = new Float64Array(n).fill(-Infinity);
    this.spikeTimes = new Float32Array(n * SPIKE_SLOTS).fill(EMPTY_SPIKE);
    this.spikeGen = new Uint8Array(n * SPIKE_SLOTS).fill(NO_GENERATION);
    this.spikeHead = new Uint8Array(n);
    this.drivenIn = new Uint8Array(n).fill(NO_GENERATION);
    this.input = new Float32Array(n);
    this.stimulated = new Uint8Array(n);
    this.pendingStimulus = new Float32Array(n);
    this.random = rng(init.seed);
    this.delay = synapseDelays(init.synapses, this.params);
    this.weight = synapseWeights(init.synapses, init.inhibitory, this.params);
    this.gain = drivenGainTable(this.params);
    this.maxDelay = new Float32Array(n);
    this.minDelay = new Float32Array(n).fill(Infinity);
    for (let s = 0; s < init.synapses.count; s++) {
      const p = init.synapses.pre[s];
      if (this.delay[s] > this.maxDelay[p]) this.maxDelay[p] = this.delay[s];
      if (this.delay[s] < this.minDelay[p]) this.minDelay[p] = this.delay[s];
    }
  }

  /** Adds `amount` to the neuron's input and fires it on the next step it is not refractory; watches its cascade. */
  stimulate(neuron: number, amount = STIMULUS): void {
    this.input[neuron] += amount;
    this.pendingStimulus[neuron] += amount;
    this.stimulated[neuron] = 1;
    this.watched = neuron;
  }

  /** Advances the model from t - dt to t. */
  step(t: number, dt: number): StepResult {
    const arrivals = this.synapsePass(t, dt);
    const spikes = this.neuronPass(t, dt);
    return { spikes, arrivals: arrivals.sample, arrivalCount: arrivals.count };
  }

  private synapsePass(t: number, dt: number) {
    const { preOffsets, post } = this.syn;
    const lo = t - dt;
    const sample: ModelArrival[] = [];
    let others = 0;
    let count = 0;
    for (let p = 0; p < this.neuronCount; p++) {
      const start = preOffsets[p];
      const end = preOffsets[p + 1];
      if (start === end) continue;
      for (let k = 0; k < SPIKE_SLOTS; k++) {
        const tau = this.spikeTimes[p * SPIKE_SLOTS + k];
        // Nothing from this spike can land in (lo, t].
        if (tau + this.maxDelay[p] <= lo || tau + this.minDelay[p] > t) continue;
        const gen = this.spikeGen[p * SPIKE_SLOTS + k];
        const gain = this.gain[gen];
        for (let s = start; s < end; s++) {
          const arrival = tau + this.delay[s];
          if (arrival <= lo || arrival > t) continue;
          const q = post[s];
          const w = this.weight[s];
          // The gain is exactly 1 for an ordinary spike, so the background is bit-identical to the ungained model.
          this.input[q] += w * gain;
          // Only an excitatory pulse can be what fired its target.
          if (gen < this.drivenIn[q] && w > 0) this.drivenIn[q] = gen;
          count++;
          if (p === this.watched || gen !== NO_GENERATION) sample.push({ pre: p, post: q, synapse: s, time: arrival });
          else if (others < ARRIVAL_SAMPLE) {
            others++;
            sample.push({ pre: p, post: q, synapse: s, time: arrival });
          }
        }
      }
    }
    return { sample, count };
  }

  private neuronPass(t: number, dt: number): ModelSpike[] {
    const p = this.params;
    const decay = Math.exp(-dt / (p.tauMs / 1000));
    const bgChance = p.backgroundRateHz > 0 ? 1 - Math.exp(-p.backgroundRateHz * dt) : 0;
    const spikes: ModelSpike[] = [];
    for (let n = 0; n < this.neuronCount; n++) {
      const drivenIn = this.drivenIn[n];
      this.drivenIn[n] = NO_GENERATION;
      if (t < this.refractoryUntil[n]) {
        // Synaptic and background input is discarded during the refractory period, but a visitor's
        // stimulus is held for the next step, so a press timed against the cell's own spike still
        // fires it once the window closes.
        this.v[n] = p.vReset;
        this.input[n] = this.stimulated[n] ? this.pendingStimulus[n] : 0;
        continue;
      }
      let input = this.input[n];
      this.input[n] = 0;
      if (bgChance > 0 && this.random() < bgChance) input += p.backgroundWeight;
      const stimulated = this.stimulated[n] === 1;
      this.stimulated[n] = 0;
      this.pendingStimulus[n] = 0;
      const v = p.vRest + (this.v[n] - p.vRest) * decay + input;
      if (v >= p.vThreshold || stimulated) {
        const head = this.spikeHead[n];
        const generation = nextGeneration(stimulated, drivenIn);
        this.spikeTimes[n * SPIKE_SLOTS + head] = t;
        this.spikeGen[n * SPIKE_SLOTS + head] = generation;
        this.spikeHead[n] = (head + 1) % SPIKE_SLOTS;
        this.v[n] = p.vReset;
        this.refractoryUntil[n] = t + p.refractoryMs / 1000;
        spikes.push({ neuron: n, time: t, stimulated, generation });
      } else {
        this.v[n] = v;
      }
    }
    return spikes;
  }
}
