import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { beforeEach, describe, expect, it } from "vitest";
import { CascadeTracker, MAX_EDGES, STALE_S, TIMELINE_S } from "./cascade";
import { EventBus } from "./events";
import { decodeNeurons, decodeSynapses, parseManifest } from "./format";
import { SpikingModel } from "./sim/model";

let bus: EventBus;
let now: number;
let tracker: CascadeTracker;

const spike = (neuron: number, time: number) => bus.emit("spike", { neuron, time, stimulated: false });
const arrive = (pre: number, post: number, time: number) => bus.emit("arrive", { pre, post, synapse: 0, time });
const stimulate = (neuron: number) => bus.emit("stimulate", { neuron });

beforeEach(() => {
  bus = new EventBus();
  now = 1;
  tracker = new CascadeTracker(bus, { clock: () => now });
});

describe("CascadeTracker", () => {
  it("counts reached cells, hop 1 and hop 2 from a scripted sequence", () => {
    stimulate(5);
    // The root's pulses land on 10, 11, 12 (12 twice) and 13.
    arrive(5, 10, 1.4);
    arrive(5, 11, 1.4);
    arrive(5, 12, 1.5);
    arrive(5, 12, 1.6);
    arrive(5, 13, 1.7);
    // 10 and 11 fire within 50 ms of their arrival; 13 fires too late; 14 was never reached.
    spike(10, 1.41);
    spike(11, 1.43);
    spike(13, 1.9);
    spike(14, 1.42);
    // 10 drives 20 and 21; 20 fires, 21 does not. 30 is driven by an unrelated cell.
    arrive(10, 20, 2.0);
    arrive(10, 21, 2.0);
    arrive(99, 30, 2.0);
    spike(20, 2.02);
    spike(30, 2.02);

    const s = tracker.summary();
    expect(s.root).toBe(5);
    expect(s.reached).toBe(4);
    expect(s.hop1).toBe(2);
    expect(s.hop2).toBe(1);
    expect(s.total).toBe(3);
    expect(s.durationS).toBeCloseTo(1.02, 6);
    expect(s.hop1SpanS).toBeCloseTo(0.43, 6);
    expect(tracker.edges()).toEqual([
      [5, 10, 1.41],
      [5, 11, 1.43],
      [10, 20, 2.02],
    ]);
    expect(tracker.latest()).toBe(20);
  });

  it("matches a spike that arrives before its arrival event, as one step reports spikes first", () => {
    stimulate(1);
    spike(2, 1.3);
    arrive(1, 2, 1.28);
    spike(3, 1.5);
    arrive(2, 3, 1.49);
    expect(tracker.summary()).toMatchObject({ reached: 1, hop1: 1, hop2: 1 });
    expect(tracker.edges()).toEqual([
      [1, 2, 1.3],
      [2, 3, 1.5],
    ]);
  });

  it("does not count a cell that fired before the pulse landed or before the stimulus", () => {
    now = 5;
    stimulate(1);
    spike(2, 4.99);
    arrive(1, 2, 5.5);
    arrive(1, 3, 4.5);
    spike(3, 4.52);
    expect(tracker.summary()).toMatchObject({ reached: 1, hop1: 0, hop2: 0 });
  });

  it("does not count a hop-1 cell again as hop 2, nor the root", () => {
    stimulate(1);
    arrive(1, 2, 1.1);
    arrive(1, 3, 1.1);
    spike(2, 1.11);
    spike(3, 1.11);
    arrive(2, 3, 1.3);
    spike(3, 1.31);
    arrive(2, 1, 1.3);
    spike(1, 1.31);
    expect(tracker.summary()).toMatchObject({ hop1: 2, hop2: 0, total: 2 });
  });

  it("ignores arrivals from inhibitory cells", () => {
    tracker.dispose();
    tracker = new CascadeTracker(bus, { clock: () => now, inhibitory: [0, 1, 0, 0] });
    stimulate(0);
    arrive(0, 2, 1.1);
    spike(2, 1.11);
    arrive(2, 3, 1.3);
    spike(3, 1.31);
    stimulate(1);
    arrive(1, 2, 1.5);
    spike(2, 1.51);
    expect(tracker.summary()).toMatchObject({ root: 1, hop1: 0 });
  });

  it("starts a fresh cascade on each stimulate", () => {
    stimulate(1);
    arrive(1, 2, 1.1);
    spike(2, 1.11);
    stimulate(7);
    expect(tracker.summary()).toEqual({ root: 7, reached: 0, hop1: 0, hop2: 0, total: 0, durationS: 0, hop1SpanS: 0 });
    expect(tracker.edges()).toEqual([]);
    expect(tracker.timeline()).toEqual([]);
    expect(tracker.latest()).toBe(-1);
  });

  it("keeps the timeline to the last 8 s, relative to the stimulus", () => {
    stimulate(1);
    arrive(1, 2, 1.1);
    spike(2, 1.11);
    arrive(2, 3, 3);
    spike(3, 3.01);
    expect(tracker.timeline().map((k) => [k.hop, k.neuron, +k.t.toFixed(2)])).toEqual([
      [1, 2, 0.11],
      [2, 3, 2.01],
    ]);
    now = 3.02 + TIMELINE_S - 0.5;
    expect(tracker.timeline().map((k) => k.neuron)).toEqual([3]);
    expect(tracker.elapsedS()).toBeCloseTo(now - 1, 9);
  });

  it("goes stale after 12 s without a new caused event, and a new event revives nothing", () => {
    expect(tracker.active).toBe(false);
    stimulate(1);
    expect(tracker.active).toBe(true);
    arrive(1, 2, 1.1);
    spike(2, 1.11);
    now = 1.11 + STALE_S - 0.01;
    expect(tracker.active).toBe(true);
    now = 1.11 + STALE_S + 0.01;
    expect(tracker.active).toBe(false);
    arrive(1, 4, now);
    expect(tracker.summary().reached).toBe(1);
    stimulate(1);
    expect(tracker.active).toBe(true);
  });

  it("caps the edges at 400, hop 1 first", () => {
    stimulate(1);
    for (let n = 10; n < 10 + MAX_EDGES + 50; n++) {
      arrive(1, n, 1.1);
      spike(n, 1.11);
    }
    expect(tracker.edges()).toHaveLength(MAX_EDGES);
    expect(tracker.summary().hop1).toBe(MAX_EDGES + 50);
    expect(tracker.edges()[0]).toEqual([1, 10, 1.11]);
  });

  it("stops listening once disposed", () => {
    tracker.dispose();
    stimulate(1);
    expect(tracker.active).toBe(false);
  });
});

describe("CascadeTracker on the real wiring", () => {
  it("counts a well-connected cell's reach and what it set off, driven by the CPU model", () => {
    const DATA = fileURLToPath(new URL("../../public/data/", import.meta.url));
    const read = (rel: string) => {
      const b = readFileSync(DATA + rel);
      return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
    };
    const manifest = parseManifest(JSON.parse(readFileSync(DATA + "manifest.json", "utf8")));
    const neurons = decodeNeurons(read(manifest.neurons));
    const synapses = decodeSynapses(read(manifest.synapses!.file));
    const out = (n: number) => synapses.preOffsets[n + 1] - synapses.preOffsets[n];
    let root = 0;
    for (let n = 0; n < neurons.count; n++) if (!neurons.inhibitory[n] && out(n) > out(root)) root = n;

    const model = new SpikingModel({ neuronCount: neurons.count, inhibitory: neurons.inhibitory, synapses, seed: 11 });
    const dt = 1 / 60;
    for (let i = 1; i <= 120; i++) model.step(-2 + i * dt, dt);
    let t = 0;
    const real = new CascadeTracker(bus, { clock: () => t, inhibitory: neurons.inhibitory });
    stimulate(root);
    model.stimulate(root);
    for (let i = 1; i <= 600; i++) {
      t = i * dt;
      const r = model.step(t, dt);
      for (const s of r.spikes) bus.emit("spike", s);
      for (const a of r.arrivals) arrive(a.pre, a.post, a.time);
    }
    const s = real.summary();
    expect(s.reached).toBeGreaterThan(50);
    expect(s.reached).toBeLessThanOrEqual(out(root));
    expect(s.hop1).toBeGreaterThan(0);
    expect(s.hop1).toBeLessThan(s.reached);
    expect(real.edges().length).toBeGreaterThanOrEqual(s.total);
    for (const [from, to, time] of real.edges()) {
      expect(time).toBeGreaterThanOrEqual(0);
      expect(from).not.toBe(to);
    }
  });
});
