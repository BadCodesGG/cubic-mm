import { describe, expect, it } from "vitest";
import { synthDataset } from "../synth";
import { SpikingModel, type ModelSynapses } from "./model";

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

function threeNeuronNet(): SpikingModel {
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
    params: { backgroundRateHz: 0, excWeight: 1.5, inhWeight: -0.5, sizeGain: 0 },
  });
}

const DT = 0.0005;

/** Runs from `from` to `to` in DT steps, collecting every spike and arrival. */
function run(model: SpikingModel, from: number, to: number) {
  const spikes: { neuron: number; time: number; stimulated: boolean }[] = [];
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
    expect(spikes.filter((s) => s.neuron === A)).toEqual([{ neuron: A, time: DT, stimulated: true }]);
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
    m.stimulate(A);
    const inside = run(m, 0.001, 0.003);
    expect(inside.spikes.filter((s) => s.neuron === A)).toHaveLength(0);
    run(m, 0.003, 0.006);
    m.stimulate(A);
    const outside = run(m, 0.006, 0.007);
    expect(outside.spikes.filter((s) => s.neuron === A)).toHaveLength(1);
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
