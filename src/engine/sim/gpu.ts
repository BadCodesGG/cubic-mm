/**
 * The GPU twin of `model.ts`: the same leaky integrate-and-fire step as three TSL compute kernels
 * on the WebGPU backend. Only construct this when `renderer.backend.isWebGPUBackend` is true.
 *
 * Per substep:
 *   reset    (1 invocation, first substep of a frame only) zeroes the event counters.
 *   synapse  (1 invocation per synapse) checks the presynaptic cell's 8 spike times for a pulse
 *            landing in (t - dt, t] and `atomicAdd`s the weight into the post cell's input, which
 *            is i32 fixed point (1 unit = 1/65536) so the sum is exact and order-independent.
 *   neuron   (1 invocation per neuron) drains its input, adds the stimulus and Poisson background
 *            (a PCG hash of seed, substep counter and neuron index), integrates, thresholds, and
 *            writes spike times into the storage buffer the ribbon material reads directly.
 *
 * Spikes and a sample of arrivals are appended to one atomic u32 event buffer, which is copied
 * to the CPU once per frame through a small pool of `ReadbackBuffer`s, so the loop never waits.
 */

import { ReadbackBuffer, StorageBufferAttribute, type WebGPURenderer } from "three/webgpu";
import {
  Fn,
  If,
  Loop,
  atomicAdd,
  atomicLoad,
  atomicStore,
  float,
  floatBitsToUint,
  instanceIndex,
  int,
  select,
  storage,
  uint,
  uniform,
} from "three/tsl";
import type { ComputeNode, Node } from "three/webgpu";
import {
  ARRIVAL_SAMPLE,
  DEFAULT_PARAMS,
  EMPTY_SPIKE,
  SPIKE_SLOTS,
  synapseDelays,
  synapseWeights,
  type ModelInit,
  type ModelParams,
} from "./model";

/** Most arrivals from the watched (visitor-stimulated) cell kept per frame. */
export const WATCH_CAP = 512;
const FIXED_ONE = 65536;
const HEADER = 4;
/** Readbacks in flight at once before a frame's events are dropped rather than queued. */
const MAX_READBACKS = 6;

export interface GpuSpike {
  neuron: number;
  time: number;
  stimulated: boolean;
}
export interface GpuArrival {
  synapse: number;
  time: number;
}
/** One frame's worth of events, a frame or two after it was simulated. */
export interface GpuEvents {
  spikes: GpuSpike[];
  arrivals: GpuArrival[];
  /** True totals, including events beyond the buffers' capacity. */
  spikeCount: number;
  arrivalCount: number;
}

/** PCG hash on u32 (pcg-random.org via Jarzynski and Olano), the same one three's `hash` uses. */
const pcg = Fn(([x]: [Node<"uint">]) => {
  const state = x.mul(747796405).add(2891336453);
  const word = state.shiftRight(state.shiftRight(28).add(4)).bitXor(state).mul(277803737);
  return word.shiftRight(22).bitXor(word);
});

export class GpuSimulation {
  readonly params: ModelParams;
  /** Spike ring, 8 floats per neuron: the ribbon, soma and sprite materials read this directly. */
  readonly spikeTimes: StorageBufferAttribute;

  private readonly renderer: WebGPURenderer;
  private readonly spikeCap: number;
  private readonly events: StorageBufferAttribute;
  private readonly stim: StorageBufferAttribute;
  private readonly stimData: Float32Array;
  private stimDirty = false;
  private readonly u = {
    t: uniform(0),
    lo: uniform(0),
    decay: uniform(1),
    bgChance: uniform(0),
    counter: uniform(0, "uint"),
    watched: uniform(0xffffffff, "uint"),
  };
  private readonly resetKernel: ComputeNode;
  private readonly synapseKernel: ComputeNode;
  private readonly neuronKernel: ComputeNode;
  private readonly pool: ReadbackBuffer[] = [];
  private readonly ready: GpuEvents[] = [];
  private inFlight = 0;
  private frameOpen = false;
  private counter = 0;
  private disposed = false;
  /** Frames whose events were dropped because every readback was still in flight. */
  dropped = 0;

  constructor(renderer: WebGPURenderer, init: ModelInit) {
    this.renderer = renderer;
    const p = (this.params = { ...DEFAULT_PARAMS, ...init.params });
    const n = init.neuronCount;
    const syn = init.synapses;
    this.spikeCap = Math.max(1024, n * 2);

    const weightQ = Int32Array.from(synapseWeights(syn, init.inhibitory, p), (w) => Math.round(w * FIXED_ONE));
    const buf = (array: Float32Array | Uint32Array | Int32Array, name: string) => {
      const a = new StorageBufferAttribute(array, 1);
      a.name = name;
      return a;
    };
    // Storage bindings must be non-empty, so a table with no synapses still gets one (dead) entry.
    const synLen = Math.max(1, syn.count);
    const pre = storage(buf(padded(Uint32Array.from(syn.pre), synLen), "simPre"), "uint", synLen).toReadOnly();
    const post = storage(buf(padded(Uint32Array.from(syn.post), synLen), "simPost"), "uint", synLen).toReadOnly();
    const delay = storage(buf(padded(synapseDelays(syn, p), synLen, 1e9), "simDelay"), "float", synLen).toReadOnly();
    const weight = storage(buf(padded(weightQ, synLen), "simWeight"), "int", synLen).toReadOnly();

    this.spikeTimes = buf(new Float32Array(n * SPIKE_SLOTS).fill(EMPTY_SPIKE), "simSpikeTimes");
    const spikes = storage(this.spikeTimes, "float", n * SPIKE_SLOTS);
    const spikesRead = storage(this.spikeTimes, "float", n * SPIKE_SLOTS).toReadOnly();
    const v = storage(buf(new Float32Array(n).fill(p.vRest), "simV"), "float", n);
    const refractory = storage(buf(new Float32Array(n).fill(-1e9), "simRefractory"), "float", n);
    const head = storage(buf(new Uint32Array(n), "simHead"), "uint", n);
    this.stimData = new Float32Array(n);
    this.stim = buf(this.stimData, "simStim");
    const stim = storage(this.stim, "float", n);
    const input = storage(buf(new Int32Array(n), "simInput"), "int", n).toAtomic();
    const eventLen = HEADER + 2 * (this.spikeCap + ARRIVAL_SAMPLE + WATCH_CAP);
    this.events = buf(new Uint32Array(eventLen), "simEvents");
    const ev = storage(this.events, "uint", eventLen).toAtomic();
    const arrivalBase = HEADER + 2 * this.spikeCap;
    const watchBase = arrivalBase + 2 * ARRIVAL_SAMPLE;
    const { u } = this;

    this.resetKernel = Fn(() => {
      for (let i = 0; i < HEADER; i++) atomicStore(ev.element(i), uint(0));
    })()
      .compute(1)
      .setName("simReset");

    this.synapseKernel = Fn(() => {
      const s = instanceIndex;
      If(s.lessThan(uint(syn.count)), () => {
        const p0 = pre.element(s).toVar();
        const q = post.element(s).toVar();
        const d = delay.element(s).toVar();
        const w = weight.element(s).toVar();
        Loop(SPIKE_SLOTS, ({ i }) => {
          const arrival = spikesRead.element(p0.mul(SPIKE_SLOTS).add(uint(i))).add(d).toVar();
          If(arrival.greaterThan(u.lo).and(arrival.lessThanEqual(u.t)), () => {
            atomicAdd(input.element(q), w);
            atomicAdd(ev.element(1), uint(1));
            const record = (counterSlot: number, base: number, cap: number) => {
              const k = atomicAdd(ev.element(counterSlot), uint(1)).toVar();
              If(k.lessThan(uint(cap)), () => {
                atomicStore(ev.element(k.mul(2).add(base)), s);
                atomicStore(ev.element(k.mul(2).add(base + 1)), floatBitsToUint(arrival));
              });
            };
            If(p0.equal(u.watched), () => record(3, watchBase, WATCH_CAP)).Else(() =>
              record(2, arrivalBase, ARRIVAL_SAMPLE),
            );
          });
        });
      });
    })()
      .compute(synLen)
      .setName("simSynapses");

    const seedHash = hashSeed(init.seed);
    this.neuronKernel = Fn(() => {
      const i = instanceIndex;
      If(i.lessThan(uint(n)), () => {
        const drive = float(atomicLoad(input.element(i))).div(FIXED_ONE).toVar();
        atomicStore(input.element(i), int(0));
        const kick = stim.element(i).toVar();
        stim.element(i).assign(0);
        drive.addAssign(kick);
        const r = pcg(pcg(u.counter.bitXor(uint(seedHash))).add(i));
        If(float(r).mul(1 / 4294967296).lessThan(u.bgChance), () => {
          drive.addAssign(p.backgroundWeight);
        });
        If(u.t.lessThan(refractory.element(i)), () => {
          v.element(i).assign(p.vReset);
        }).Else(() => {
          const next = v.element(i).sub(p.vRest).mul(u.decay).add(p.vRest).add(drive).toVar();
          If(next.greaterThanEqual(p.vThreshold), () => {
            const h = head.element(i).toVar();
            spikes.element(i.mul(SPIKE_SLOTS).add(h)).assign(u.t);
            head.element(i).assign(h.add(1).bitAnd(SPIKE_SLOTS - 1));
            v.element(i).assign(p.vReset);
            refractory.element(i).assign(u.t.add(p.refractoryMs / 1000));
            const k = atomicAdd(ev.element(0), uint(1)).toVar();
            If(k.lessThan(uint(this.spikeCap)), () => {
              const flag = select(kick.greaterThan(0), uint(0x80000000), uint(0));
              atomicStore(ev.element(k.mul(2).add(HEADER)), i.bitOr(flag));
              atomicStore(ev.element(k.mul(2).add(HEADER + 1)), floatBitsToUint(u.t));
            });
          }).Else(() => {
            v.element(i).assign(next);
          });
        });
      });
    })()
      .compute(n)
      .setName("simNeurons");
  }

  stimulate(neuron: number, amount: number): void {
    this.stimData[neuron] += amount;
    this.stimDirty = true;
    this.u.watched.value = neuron;
  }

  /** Runs one substep from t - dt to t. Call `endFrame` after the frame's last substep. */
  substep(t: number, dt: number): void {
    const { u, params: p } = this;
    u.t.value = t;
    u.lo.value = t - dt;
    u.decay.value = Math.exp(-dt / (p.tauMs / 1000));
    u.bgChance.value = p.backgroundRateHz > 0 ? 1 - Math.exp(-p.backgroundRateHz * dt) : 0;
    u.counter.value = this.counter++ >>> 0;
    if (this.stimDirty) {
      this.stim.needsUpdate = true;
    }
    const kernels = this.frameOpen ? [this.synapseKernel, this.neuronKernel] : [this.resetKernel, this.synapseKernel, this.neuronKernel];
    this.frameOpen = true;
    this.renderer.compute(kernels);
    if (this.stimDirty) {
      // The upload happened inside compute(); the kernel zeroes its copy as it reads it.
      this.stimData.fill(0);
      this.stimDirty = false;
    }
  }

  /** Queues a copy of this frame's events to the CPU. Results appear in `drain()` later. */
  endFrame(): void {
    if (!this.frameOpen) return;
    this.frameOpen = false;
    let target = this.pool.pop();
    if (!target) {
      if (this.inFlight >= MAX_READBACKS) {
        this.dropped++;
        return;
      }
      target = new ReadbackBuffer(this.events.array.byteLength);
      target.name = "simEventsReadback";
    }
    this.inFlight++;
    const rb = target;
    this.renderer
      .getArrayBufferAsync(this.events, rb)
      .then(() => {
        if (!this.disposed && rb.buffer) this.ready.push(this.parse(new Uint32Array(rb.buffer)));
        rb.release();
        if (this.disposed) rb.dispose();
        else this.pool.push(rb);
      })
      .catch(() => rb.dispose())
      .finally(() => this.inFlight--);
  }

  /** Every frame's events that have come back since the last call, oldest first. */
  drain(): GpuEvents[] {
    return this.ready.splice(0);
  }

  /** Resolves once every queued readback has landed. */
  async settle(): Promise<void> {
    while (this.inFlight > 0) await new Promise((r) => setTimeout(r, 5));
  }

  private parse(words: Uint32Array): GpuEvents {
    const f32 = new Float32Array(words.buffer, words.byteOffset, words.length);
    const spikeCount = words[0];
    const arrivalCount = words[1];
    const spikes: GpuSpike[] = [];
    for (let k = 0; k < Math.min(spikeCount, this.spikeCap); k++) {
      const id = words[HEADER + 2 * k];
      spikes.push({ neuron: id & 0x7fffffff, stimulated: id >>> 31 === 1, time: f32[HEADER + 2 * k + 1] });
    }
    const arrivals: GpuArrival[] = [];
    const watchBase = HEADER + 2 * (this.spikeCap + ARRIVAL_SAMPLE);
    for (let k = 0; k < Math.min(words[3], WATCH_CAP); k++) {
      arrivals.push({ synapse: words[watchBase + 2 * k], time: f32[watchBase + 2 * k + 1] });
    }
    const arrivalBase = HEADER + 2 * this.spikeCap;
    for (let k = 0; k < Math.min(words[2], ARRIVAL_SAMPLE); k++) {
      arrivals.push({ synapse: words[arrivalBase + 2 * k], time: f32[arrivalBase + 2 * k + 1] });
    }
    return { spikes, arrivals, spikeCount, arrivalCount };
  }

  dispose(): void {
    this.disposed = true;
    for (const rb of this.pool.splice(0)) rb.dispose();
    for (const k of [this.resetKernel, this.synapseKernel, this.neuronKernel]) k.dispose();
  }
}

function padded<T extends Float32Array | Uint32Array | Int32Array>(a: T, length: number, fill = 0): T {
  if (a.length >= length) return a;
  const out = new (a.constructor as new (n: number) => T)(length);
  out.fill(fill);
  out.set(a);
  return out;
}

/** Folds the seed into one u32 on the CPU, so the GPU stream depends on it bit for bit. */
function hashSeed(seed: number): number {
  let h = Math.imul(seed >>> 0, 0x9e3779b1) >>> 0;
  h ^= h >>> 16;
  return h >>> 0;
}
