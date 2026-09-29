import { describe, expect, it } from "vitest";
import { synthDataset } from "../synth";
import { NO_GENERATION, SpikingModel, nextGeneration, type ModelParams, type ModelSpike, type ModelSynapses } from "./model";

/** Recorded from the run below; a change here means the model or the synthetic wiring changed. */
const GOLDEN_SPIKES = 521;

/**
 * A -> B excitatory, B -> C inhibitory. Path distances are quarter-µm: A's synapse sits 100 µm
 * down its axon, B's 200 µm. Weights are raised so one input is enough to fire (or clearly
 * suppress) the target, and background drive is off, so every spike here is caused.
 */
const A = 0;
const B = 1;
const C = 2;

function threeNeuronNet(params: Partial<ModelParams> = {}): SpikingModel {
  const synapses: ModelSynapses = {
    count: 2,
    preOffsets: new Uint32Array([0, 1, 2, 2]),
    pre: new Uint16Array([A, B]),
    post: new Uint16Array([B, C]),
    preDistQ: new Uint16Array([400, 800]),
    size: new Uint8Array([0, 0]),
  };
  return new SpikingModel({
    neuronCount: 3,
    inhibitory: new Uint8Array([0, 1, 0]),
    synapses,
    seed: 1,
    params: { backgroundRateHz: 0, excWeight: 1.5, inhWeight: -0.5, sizeGain: 0, ...params },
  });
}

const DT = 0.0005;

/** Runs from `from` to `to` in DT steps, collecting every spike and arrival. */
function run(model: SpikingModel, from: number, to: number) {
  const spikes: ModelSpike[] = [];
  const arrivals: { pre: number; post: number; time: number }[] = [];
  const steps = Math.round((to - from) / DT);
  for (let i = 1; i <= steps; i++) {
    const r = model.step(from + i * DT, DT);
    spikes.push(...r.spikes);
    arrivals.push(...r.arrivals);
  }
  return { spikes, arrivals };
}

describe("SpikingModel on a hand-built three-neuron net", () => {
  it("fires a stimulated neuron exactly once", () => {
    const m = threeNeuronNet();
    m.stimulate(A);
    const { spikes } = run(m, 0, 0.1);
    expect(spikes.filter((s) => s.neuron === A)).toEqual([{ neuron: A, time: DT, stimulated: true, generation: 0 }]);
  });

  it("fires B one conduction delay after A: 100 µm / 0.5 m/s x 1000 slow-mo + 1 ms synaptic delay", () => {
    const m = threeNeuronNet();
    m.stimulate(A);
    const { spikes, arrivals } = run(m, 0, 0.5);
    const a = spikes.find((s) => s.neuron === A)!;
    const b = spikes.find((s) => s.neuron === B)!;
    expect(b).toBeDefined();
    expect(b.stimulated).toBe(false);
    // 100e-6 m / 0.5 m/s = 0.2 ms, x1000 = 0.2 s, plus 1 ms.
    const expected = a.time + 0.201;
    expect(b.time).toBeGreaterThanOrEqual(expected - 1e-9);
    // Within one step: the arrival lands on a step boundary here, so float rounding decides which side.
    expect(b.time - expected).toBeLessThanOrEqual(DT + 1e-9);
    expect(arrivals).toContainEqual(expect.objectContaining({ pre: A, post: B }));
  });

  it("pushes C's membrane potential down when B's inhibitory pulse arrives, and C does not fire", () => {
    const m = threeNeuronNet();
    m.stimulate(A);
    const { spikes } = run(m, 0, 0.5);
    const b = spikes.find((s) => s.neuron === B)!;
    // B's pulse reaches C after 200 µm: 0.4 s + 1 ms.
    const arrival = b.time + 0.401;
    const before = run(m, 0.5, arrival - 2 * DT);
    expect(m.v[C]).toBeCloseTo(0, 6);
    const after = run(m, arrival - 2 * DT, arrival + 2 * DT);
    expect(m.v[C]).toBeLessThan(-0.4);
    expect([...before.spikes, ...after.spikes].some((s) => s.neuron === C)).toBe(false);
  });

  it("refractory period blocks a second spike within 4 ms, and allows one after", () => {
    const m = threeNeuronNet();
    m.stimulate(A);
    const first = run(m, 0, 0.001);
    expect(first.spikes.filter((s) => s.neuron === A)).toHaveLength(1);
    run(m, 0.001, 0.003);
    run(m, 0.003, 0.006);
    m.stimulate(A);
    const outside = run(m, 0.006, 0.007);
    expect(outside.spikes.filter((s) => s.neuron === A)).toHaveLength(1);
  });

  it("holds a stimulus that lands inside the refractory window and fires it once the window closes", () => {
    const m = threeNeuronNet();
    m.stimulate(A);
    run(m, 0, 0.001);
    // Pressed while refractory (until 0.005): nothing fires inside the window...
    m.stimulate(A);
    const inside = run(m, 0.001, 0.004);
    expect(inside.spikes.filter((s) => s.neuron === A)).toHaveLength(0);
    // ...and the held stimulus fires on the first step after it, flagged as stimulated.
    const after = run(m, 0.004, 0.008);
    const held = after.spikes.filter((s) => s.neuron === A);
    expect(held).toHaveLength(1);
    expect(held[0].stimulated).toBe(true);
    // The first spike was at 0.0005, so the window closes at 0.0045 and the held stimulus fires there.
    expect(held[0].time).toBeCloseTo(0.0045, 6);
  });

  it("fires a stimulated neuron even when inhibition holds it far below rest", () => {
    const m = threeNeuronNet();
    m.input[C] -= 3;
    run(m, 0, DT);
    expect(m.v[C]).toBeLessThan(-2);
    m.stimulate(C);
    expect(run(m, DT, 2 * DT).spikes).toEqual([{ neuron: C, time: 2 * DT, stimulated: true, generation: 0 }]);
  });

  it("records spike times in an 8-slot ring per neuron", () => {
    const m = threeNeuronNet();
    m.stimulate(A);
    run(m, 0, 0.001);
    expect(m.spikeTimes[A * 8]).toBeCloseTo(DT, 9);
    expect(Array.from(m.spikeTimes.subarray(A * 8 + 1, A * 8 + 8))).toEqual(new Array(7).fill(-1e9));
    expect(m.spikeHead[A]).toBe(1);
  });
});

describe("SpikingModel's driven cascade", () => {
  // At 0.2 one A -> B pulse cannot fire B on its own; at the stimulated spike's 8x it can.
  const weak = { excWeight: 0.2 };

  it("a stimulated A fires B (generation 1), though a plain A spike does not", () => {
    const plain = threeNeuronNet(weak);
    plain.input[A] += 2; // ordinary input, not the visitor's stimulus
    const p = run(plain, 0, 0.5);
    expect(p.spikes.filter((s) => s.neuron === A)).toEqual([{ neuron: A, time: DT, stimulated: false, generation: 255 }]);
    expect(p.spikes.some((s) => s.neuron === B)).toBe(false);

    const driven = threeNeuronNet(weak);
    driven.stimulate(A);
    const d = run(driven, 0, 0.5);
    expect(d.spikes.find((s) => s.neuron === A)).toMatchObject({ stimulated: true, generation: 0 });
    expect(d.spikes.find((s) => s.neuron === B)).toMatchObject({ stimulated: false, generation: 1 });
  });

  it("a driven B (generation 1, gain 4) inhibits C four times harder than a plain B spike", () => {
    /** C's lowest membrane potential over the run. */
    const trough = (m: SpikingModel) => {
      let low = 0;
      for (let i = 1; i <= 1400; i++) {
        m.step(i * DT, DT);
        low = Math.min(low, m.v[C]);
      }
      return low;
    };
    const plain = threeNeuronNet();
    plain.input[B] += 2;
    expect(trough(plain)).toBeCloseTo(-0.5, 5);
    const driven = threeNeuronNet();
    driven.stimulate(A);
    expect(trough(driven)).toBeCloseTo(-2, 5);
  });

  it("decays hop by hop through gains 12, 4, 2, 1, 1 down a chain", () => {
    // Cells 0..5 form a chain whose links (size 255: 0.05 x 21 = 1.05) fire the next cell unaided.
    // Each chain cell i also reaches a probe, cell 6 + i, through a size-0 synapse of 0.05, so the
    // probe's peak potential is 0.05 x the gain of cell i's generation.
    const pre: number[] = [];
    const post: number[] = [];
    const size: number[] = [];
    const preOffsets = [0];
    for (let i = 0; i < 6; i++) {
      if (i < 5) {
        pre.push(i);
        post.push(i + 1);
        size.push(255);
      }
      pre.push(i);
      post.push(6 + i);
      size.push(0);
      preOffsets.push(pre.length);
    }
    for (let i = 6; i < 12; i++) preOffsets.push(pre.length);
    const m = new SpikingModel({
      neuronCount: 12,
      inhibitory: new Uint8Array(12),
      synapses: {
        count: pre.length,
        preOffsets: new Uint32Array(preOffsets),
        pre: new Uint16Array(pre),
        post: new Uint16Array(post),
        preDistQ: new Uint16Array(pre.length).fill(4),
        size: new Uint8Array(size),
      },
      seed: 1,
      params: { backgroundRateHz: 0, excWeight: 0.05, sizeGain: 20 },
    });
    m.stimulate(0);
    const peak = new Array(6).fill(0);
    const gens = new Map<number, number>();
    for (let i = 1; i <= 200; i++) {
      for (const s of m.step(i * DT, DT).spikes) gens.set(s.neuron, s.generation);
      for (let k = 0; k < 6; k++) peak[k] = Math.max(peak[k], m.v[6 + k]);
    }
    expect([0, 1, 2, 3, 4, 5].map((n) => gens.get(n))).toEqual([0, 1, 2, 3, 4, 5]);
    expect(peak.map((v) => +(v / 0.05).toFixed(4))).toEqual([12, 4, 2, 1, 1, 1]);
  });

  it("caps generations at NO_GENERATION, which an ordinary spike keeps", () => {
    expect(nextGeneration(true, NO_GENERATION)).toBe(0);
    expect(nextGeneration(false, 3)).toBe(4);
    expect(nextGeneration(false, 254)).toBe(NO_GENERATION);
    expect(nextGeneration(false, NO_GENERATION)).toBe(NO_GENERATION);
  });
});

describe("SpikingModel on the 200-neuron synthetic volume", () => {
  it("produces the recorded spike count over 5 simulated seconds at a fixed seed and 60 Hz steps", () => {
    const data = synthDataset(1, 200);
    const m = new SpikingModel({
      neuronCount: data.neurons.count,
      inhibitory: data.neurons.inhibitory,
      synapses: data.synapses!,
      seed: 7,
    });
    const dt = 1 / 60;
    let spikes = 0;
    let arrivals = 0;
    for (let i = 1; i <= 300; i++) {
      const r = m.step(i * dt, dt);
      spikes += r.spikes.length;
      arrivals += r.arrivalCount;
    }
    console.log(`golden: ${data.synapses!.count} synapses, ${spikes} spikes, ${arrivals} arrivals in 5 s`);
    expect(spikes).toBe(GOLDEN_SPIKES);
  }, 60_000);
});
