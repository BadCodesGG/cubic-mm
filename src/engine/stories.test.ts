import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { loadDataset, type Dataset } from "./data";
import { Compartment, NO_PARENT, PATH_DIST_UNIT_UM, quantise, type Manifest } from "./format";
import { Vector3 } from "three/webgpu";
import type { Pose } from "./camera/goto";
import { EventBus } from "./events";
import {
  StoryPlayer,
  buildStories,
  orbitPose,
  pickBasketCell,
  pickFeedforward,
  pickHub,
  pickLongestAxon,
  type OrbitAction,
  type StoryDef,
  type StoryStep,
} from "./stories";

/**
 * Six cells, bounds 0..1000 µm. Every synapse is listed by hand, so each pick has a worked answer.
 *
 *   0 BC   -> 2 (23P) x2, 3 (4P) x1          3 synapses, all onto excitatory cells
 *   1 BC   -> 0 (BC) x3, 2 (23P) x1          more synapses, but only 1 onto an excitatory cell
 *   2 23P  (no outputs)
 *   3 4P   -> 2 (23P) x2, 4 (23P) x1         3 synapses onto 2 layer 2/3 cells
 *   4 23P  (no outputs)
 *   5 4P   -> 2 (23P) x1, 3 (4P) x3          1 synapse onto layer 2/3
 *
 * In + out: 0: 3+3=6, 1: 0+4=4, 2: 6+0=6, 3: 4+3=7, 4: 1+0=1, 5: 0+4=4, so the hub is 3.
 * Longest axon (soma to farthest axon tip): 4 has 500 µm, the rest less.
 */
const TYPES = ["23P", "4P", "BC"];
const CELLS: { type: string; soma: [number, number, number]; axonUm: number }[] = [
  { type: "BC", soma: [100, 300, 100], axonUm: 200 },
  { type: "BC", soma: [200, 300, 100], axonUm: 250 },
  { type: "23P", soma: [100, 250, 100], axonUm: 120 },
  { type: "4P", soma: [300, 420, 100], axonUm: 300 },
  { type: "23P", soma: [150, 240, 400], axonUm: 500 },
  { type: "4P", soma: [320, 430, 100], axonUm: 90 },
];
const SYNAPSES: [pre: number, post: number][] = [
  [0, 2], [0, 2], [0, 3],
  [1, 0], [1, 0], [1, 0], [1, 2],
  [3, 2], [3, 2], [3, 4],
  [5, 2], [5, 3], [5, 3], [5, 3],
];

function fixture(): Dataset {
  const MAX = 1000;
  const q = (v: number) => quantise(v, 0, MAX);
  const n = CELLS.length;
  // Per cell: a soma, one axon node at its axon length, one dendrite node.
  const pos = new Uint16Array(n * 3 * 3);
  const parent = new Uint32Array(n * 3);
  const pathDistQ = new Uint16Array(n * 3);
  const compartment = new Uint8Array(n * 3);
  const neuronOfNode = new Uint16Array(n * 3);
  CELLS.forEach((c, i) => {
    const [x, y, z] = c.soma;
    const nodes: [number, number, number][] = [[x, y, z], [x + 20, y, z], [x, y - 20, z]];
    nodes.forEach((p, k) => p.forEach((v, a) => (pos[(i * 3 + k) * 3 + a] = q(v))));
    parent.set([NO_PARENT, i * 3, i * 3], i * 3);
    pathDistQ.set([0, c.axonUm / PATH_DIST_UNIT_UM, 20 / PATH_DIST_UNIT_UM], i * 3);
    compartment.set([Compartment.Soma, Compartment.Axon, Compartment.Dendrite], i * 3);
    neuronOfNode.fill(i, i * 3, i * 3 + 3);
  });
  const preOffsets = new Uint32Array(n + 1);
  for (const [pre] of SYNAPSES) preOffsets[pre + 1]++;
  for (let i = 0; i < n; i++) preOffsets[i + 1] += preOffsets[i];
  const count = SYNAPSES.length;
  const manifest: Manifest = {
    version: 1,
    boundsUm: { min: [0, 0, 0], max: [MAX, MAX, MAX] },
    neuronCount: n,
    cellTypes: TYPES,
    neurons: "neurons.bin",
    lods: { hi: { nodeCount: n * 3, chunks: [] }, lite: { nodeCount: n * 3, chunks: [] } },
    synapses: { file: "synapses.bin", count },
    credits: { dataset: "fixture", licence: "CC BY 4.0", url: "https://example.org", citations: ["A citation."] },
  };
  return {
    manifest,
    lod: "hi",
    neurons: {
      count: n,
      rootId: BigUint64Array.from(CELLS.map((_, i) => 864691135000000000n + BigInt(i))),
      somaUm: Float32Array.from(CELLS.flatMap((c) => c.soma)),
      cellType: Uint8Array.from(CELLS.map((c) => TYPES.indexOf(c.type))),
      inhibitory: Uint8Array.from(CELLS.map((c) => (c.type === "BC" ? 1 : 0))),
      layer: Uint8Array.from(CELLS.map((c) => (c.type === "4P" ? 4 : 2))),
    },
    nodes: { count: n * 3, pos, radiusNm: new Uint16Array(n * 3).fill(500), parent, pathDistQ, compartment, neuronOfNode },
    neuronNodeStart: Uint32Array.from(CELLS.map((_, i) => i * 3)),
    neuronNodeCount: new Uint32Array(n).fill(3),
    synapses: {
      count,
      neuronCount: n,
      preOffsets,
      pre: Uint16Array.from(SYNAPSES.map(([p]) => p)),
      post: Uint16Array.from(SYNAPSES.map(([, p]) => p)),
      pos: new Uint16Array(count * 3),
      preDistQ: Uint16Array.from(SYNAPSES.map((_, i) => (10 + i) / PATH_DIST_UNIT_UM)),
      size: new Uint8Array(count).fill(128),
    },
  };
}

describe("story picks on a worked fixture", () => {
  const data = fixture();

  it("the basket cell is the BC with the most synapses onto excitatory cells, not the most synapses", () => {
    expect(pickBasketCell(data)).toEqual({ neuron: 0, synapses: 3, targets: 2, excitatorySynapses: 3 });
  });

  it("the longest axon is the cell whose farthest axon tip is farthest along the tree", () => {
    expect(pickLongestAxon(data)).toEqual({ neuron: 4, lengthUm: 500 });
  });

  it("the feedforward cell is the 4P with the most synapses onto 23P cells", () => {
    expect(pickFeedforward(data)).toEqual({ neuron: 3, synapses: 3, targets: 2 });
  });

  it("the hub is the cell with the most synapses in and out, with its distinct partners and reach", () => {
    // Partners of 3: 0 (in), 5 (in), 2 (out), 4 (out). Farthest partner soma: 4, at |(150,240,400)-(300,420,100)|.
    expect(pickHub(data)).toEqual({ neuron: 3, inputs: 4, outputs: 3, partners: 4, reachUm: expect.closeTo(Math.hypot(150, 180, 300), 3) });
  });

  it("returns null for a pick the data cannot make", () => {
    const none = fixture();
    none.synapses = null;
    expect(pickBasketCell(none)).toBeNull();
    expect(pickFeedforward(none)).toBeNull();
    expect(pickHub(none)).toBeNull();
    none.nodes.compartment.fill(Compartment.Dendrite);
    expect(pickLongestAxon(none)).toBeNull();
  });
});

describe("story picks on the committed data", () => {
  const DATA = fileURLToPath(new URL("../../public/data/", import.meta.url));
  let data: Dataset;
  beforeAll(async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => new Response(new Uint8Array(readFileSync(DATA + url.slice("/data/".length))))),
    );
    data = await loadDataset("/data", "hi");
  }, 60_000);
  afterAll(() => vi.unstubAllGlobals());

  /** Brute force over the flat synapse arrays, independent of the offsets the picks use. */
  function tally(keep: (pre: number, post: number) => boolean): Map<number, number> {
    const syn = data.synapses!;
    const m = new Map<number, number>();
    for (let s = 0; s < syn.count; s++) if (keep(syn.pre[s], syn.post[s])) m.set(syn.pre[s], (m.get(syn.pre[s]) ?? 0) + 1);
    return m;
  }
  const typeOf = (n: number) => data.manifest.cellTypes[data.neurons.cellType[n]];

  it("picks the BC with the most synapses onto excitatory cells", () => {
    const pick = pickBasketCell(data)!;
    expect(typeOf(pick.neuron)).toBe("BC");
    const counts = tally((pre, post) => typeOf(pre) === "BC" && !data.neurons.inhibitory[post]);
    expect(pick.excitatorySynapses).toBe(Math.max(...counts.values()));
    expect(counts.get(pick.neuron)).toBe(pick.excitatorySynapses);
    expect(pick.targets).toBeGreaterThan(1);
  });

  it("picks the longest axon in the volume", () => {
    const pick = pickLongestAxon(data)!;
    let longest = 0;
    for (let i = 0; i < data.nodes.count; i++) {
      if (data.nodes.compartment[i] === Compartment.Axon) longest = Math.max(longest, data.nodes.pathDistQ[i] * PATH_DIST_UNIT_UM);
    }
    expect(pick.lengthUm).toBe(longest);
    // The owner of that tip is the pick.
    let owner = -1;
    for (let i = 0; i < data.nodes.count && owner < 0; i++) {
      if (data.nodes.compartment[i] === Compartment.Axon && data.nodes.pathDistQ[i] * PATH_DIST_UNIT_UM === longest) owner = data.nodes.neuronOfNode[i];
    }
    expect(pick.neuron).toBe(owner);
  });

  it("picks the 4P with the most synapses onto 23P cells", () => {
    const pick = pickFeedforward(data)!;
    expect(typeOf(pick.neuron)).toBe("4P");
    const counts = tally((pre, post) => typeOf(pre) === "4P" && typeOf(post) === "23P");
    expect(pick.synapses).toBe(Math.max(...counts.values()));
    expect(counts.get(pick.neuron)).toBe(pick.synapses);
  });

  it("picks the cell with the most synapses in and out", () => {
    const pick = pickHub(data)!;
    const out = tally(() => true);
    const inc = new Map<number, number>();
    const syn = data.synapses!;
    for (let s = 0; s < syn.count; s++) inc.set(syn.post[s], (inc.get(syn.post[s]) ?? 0) + 1);
    let best = 0;
    for (let n = 0; n < data.neurons.count; n++) best = Math.max(best, (out.get(n) ?? 0) + (inc.get(n) ?? 0));
    expect(pick.inputs + pick.outputs).toBe(best);
    expect(pick.inputs).toBe(inc.get(pick.neuron) ?? 0);
    expect(pick.outputs).toBe(out.get(pick.neuron) ?? 0);
  });
});

describe("StoryPlayer", () => {
  const pose = (x: number): Pose => ({ position: new Vector3(x, 0, 0), target: new Vector3(x, 0, 1) });
  const REST = pose(-1);

  function setup(steps: StoryStep[], seconds = 10) {
    const bus = new EventBus();
    const log: string[] = [];
    bus.on("story", (e) => log.push(`story:${e.id}`));
    bus.on("stimulate", (e) => log.push(`stimulate:${e.neuron}`));
    bus.on("ride", (e) => log.push(`ride:${e.neuron}`));
    bus.on("partners", (e) => log.push(`partners:${e.neuron}:${e.show}`));
    const flights: { x: number; seconds: number; at: number }[] = [];
    const placed: Vector3[] = [];
    const ends: string[] = [];
    let clock = 0;
    const player = new StoryPlayer({
      bus,
      fly: (p, s) => flights.push({ x: p.position.x, seconds: s, at: clock }),
      place: (p) => placed.push(p.position.clone()),
      rest: () => REST,
      onEnd: (id, reason) => ends.push(`${id}:${reason}`),
    });
    const def: StoryDef = { id: "t", title: "T", blurb: "b", neuron: 3, seconds, steps };
    /** Advances on a fake 60 Hz clock. */
    const run = (s: number) => {
      for (let i = 0; i < Math.round(s * 60); i++) {
        clock += 1 / 60;
        player.update(1 / 60);
      }
    };
    return { bus, log, flights, placed, ends, player, def, run };
  }

  it("runs each step once at its time, captions inside their windows, and ends on time", () => {
    const { log, flights, ends, player, def, run } = setup([
      { at: 0, do: { kind: "fly", pose: pose(5), seconds: 3 } },
      { at: 0.5, do: { kind: "caption", text: "hello", seconds: 2 } },
      { at: 4, do: { kind: "stimulate", neuron: 3 } },
    ]);
    player.start(def);
    expect(player.running).toBe(true);
    expect(log).toEqual(["story:t"]);
    expect(flights).toEqual([{ x: 5, seconds: 3, at: 0 }]);
    expect(player.caption()).toBeNull();
    run(1);
    expect(player.caption()).toBe("hello");
    run(2);
    expect(player.caption()).toBeNull();
    expect(log).toEqual(["story:t"]);
    run(1.1);
    expect(log).toEqual(["story:t", "stimulate:3"]);
    run(10);
    expect(player.running).toBe(false);
    expect(log).toEqual(["story:t", "stimulate:3", "story:null"]);
    expect(ends).toEqual(["t:done"]);
  });

  it("reads a computed caption when it is shown, not when the story is built", () => {
    let fired = 0;
    const { player, def, run } = setup([{ at: 2, do: { kind: "caption", text: () => `${fired} fired`, seconds: 2 } }]);
    player.start(def);
    run(1);
    fired = 7;
    run(1.5);
    expect(player.caption()).toBe("7 fired");
    fired = 9;
    run(0.5);
    expect(player.caption()).toBe("7 fired");
  });

  it("a ride asks for the ride before firing the cell, so the camera catches the spike", () => {
    const { log, player, def, run } = setup([{ at: 1, do: { kind: "ride", neuron: 4 } }]);
    player.start(def);
    run(1.1);
    expect(log).toEqual(["story:t", "ride:4", "stimulate:4"]);
  });

  it("a wait holds the timeline until its event, then carries on from there", () => {
    const { bus, flights, player, def, run } = setup([
      { at: 1, do: { kind: "wait", event: "rideJump", timeout: 8 } },
      { at: 1.5, do: { kind: "fly", pose: pose(9), seconds: 2 } },
    ]);
    player.start(def);
    run(4);
    expect(player.time).toBeCloseTo(1, 5);
    expect(player.elapsed).toBeCloseTo(4, 5);
    expect(flights).toEqual([]);
    bus.emit("rideJump", { fromNeuron: 3, neuron: 5, time: 1 });
    run(0.4);
    expect(flights).toEqual([]);
    run(0.2);
    expect(flights.map((f) => f.x)).toEqual([9]);
    expect(flights[0].at).toBeCloseTo(4.5, 1);
  });

  it("a wait gives up after its timeout", () => {
    const { flights, player, def, run } = setup([
      { at: 1, do: { kind: "wait", event: "rideJump", timeout: 3 } },
      { at: 1, do: { kind: "fly", pose: pose(9), seconds: 2 } },
    ]);
    player.start(def);
    run(3.9);
    expect(flights).toEqual([]);
    run(0.2);
    expect(flights.map((f) => f.x)).toEqual([9]);
  });

  it("an orbit places the camera round a full circle at its radius over its seconds", () => {
    const centre = new Vector3(100, 200, 300);
    const orbit: StoryStep = { at: 1, do: { kind: "orbit", centre, radiusUm: 50, liftUm: 10, fromAngle: 0, seconds: 15 } };
    const { placed, player, def, run } = setup([orbit], 20);
    player.start(def);
    run(1 + 15 + 1);
    expect(placed.length).toBeGreaterThan(15 * 60 - 2);
    for (const p of placed) {
      expect(Math.hypot(p.x - centre.x, p.z - centre.z)).toBeCloseTo(50, 3);
      expect(p.y).toBeCloseTo(centre.y - 10, 3);
    }
    const first = placed[0];
    const quarter = placed[Math.round(placed.length / 4)];
    const last = placed[placed.length - 1];
    expect(first.distanceTo(last)).toBeLessThan(0.5);
    expect(first.distanceTo(quarter)).toBeGreaterThan(60);
    // The pose the builder flies to before it starts is where the orbit begins.
    expect(orbitPose(orbit.do as OrbitAction, 0, pose(0)).position.distanceTo(first)).toBeLessThan(0.5);
  });

  it("a skip flies to rest in a second, ends at once, never runs later steps, and takes the wiring down", () => {
    const { log, flights, ends, player, def, run } = setup([
      { at: 0.5, do: { kind: "partners", neuron: 3, show: true } },
      { at: 5, do: { kind: "stimulate", neuron: 3 } },
    ]);
    player.start(def);
    run(1);
    player.skip();
    expect(player.running).toBe(false);
    expect(flights.at(-1)).toMatchObject({ x: -1, seconds: 1 });
    run(10);
    player.skip();
    expect(log).toEqual(["story:t", "partners:3:true", "partners:3:false", "story:null"]);
    expect(ends).toEqual(["t:skip"]);
  });

  it("starting another story ends the one playing", () => {
    const { log, player, def, run } = setup([]);
    player.start(def);
    run(1);
    player.start({ ...def, id: "u" });
    expect(log).toEqual(["story:t", "story:null", "story:u"]);
    expect(player.id).toBe("u");
  });
});

describe("buildStories", () => {
  const data = fixture();
  const live = { reached: () => 11, cascade: () => ({ hop1Cells: [2, 5], total: 6 }) };
  const stories = buildStories(data, live);
  const byId = new Map(stories.map((s) => [s.id, s]));
  /** Every caption a story shows, computed ones read now. */
  const captions = (id: string) =>
    byId.get(id)!.steps.flatMap((s) => (s.do.kind === "caption" ? [typeof s.do.text === "function" ? s.do.text() : s.do.text] : []));

  it("builds the four stories round the picked cells, each 25 to 45 s with its steps in order", () => {
    expect(stories.map((s) => [s.id, s.neuron])).toEqual([["basket", 0], ["axon", 4], ["layers", 3], ["hub", 3]]);
    for (const s of stories) {
      expect(s.seconds).toBeGreaterThanOrEqual(25);
      expect(s.seconds).toBeLessThanOrEqual(45);
      expect(s.steps.map((x) => x.at)).toEqual([...s.steps.map((x) => x.at)].sort((a, b) => a - b));
      expect(s.steps.at(-1)!.at).toBeLessThan(s.seconds);
      expect(s.title.length).toBeGreaterThan(0);
      expect(s.blurb.length).toBeGreaterThan(0);
    }
  });

  it("quotes the basket cell's measured counts, and how many cells its pulse reached", () => {
    expect(captions("basket")[0]).toContain("3 synapses onto 2 neighbouring cells");
    expect(captions("basket")).toContain("11 cells just received a pulse that pushes them further from firing.");
    expect(captions("basket")[1]).toContain("Slowed 5 times more");
    expect(byId.get("basket")!.steps.filter((s) => s.do.kind === "stimulate").length).toBeGreaterThanOrEqual(1);
  });

  it("quotes the axon's length and its travel time at the model's speed", () => {
    // 500 µm at 0.5 m/s shown 1000x slower: 1.0 s.
    expect(captions("axon")[0]).toContain("500 µm");
    expect(captions("axon").join(" ")).toContain("One spike, 500 µm of wire, 1.0 seconds at 1,000x slower than life");
    // Cell 4 makes no synapses, so there is no count to quote.
    expect(captions("axon").join(" ")).not.toContain("Along the way");
    const kinds = byId.get("axon")!.steps.map((s) => s.do.kind);
    expect(kinds.indexOf("ride")).toBeLessThan(kinds.indexOf("wait"));
  });

  it("counts only the layer 2/3 targets among the cells that fired", () => {
    expect(captions("layers")[0]).toContain("sends 3 synapses up to 2 layer 2/3 cells");
    // Of 3's 23P targets (2 and 4), only 2 is in hop 1; 5 fired but is not a target.
    expect(captions("layers")).toContain("1 of those 2 layer 2/3 cells fired.");
  });

  it("orbits the hub once over 15 s with its wiring up for exactly the orbit", () => {
    const steps = byId.get("hub")!.steps;
    const orbit = steps.find((s) => s.do.kind === "orbit")!;
    expect((orbit.do as OrbitAction).seconds).toBe(15);
    const wiring = steps.filter((s) => s.do.kind === "partners");
    expect(wiring.map((s) => [s.at, (s.do as { show: boolean }).show])).toEqual([[orbit.at, true], [orbit.at + 15, false]]);
    expect(captions("hub")[0]).toContain("talks to 4 other cells: 4 synapses in, 3 out");
  });

  it("leaves out a story the data cannot tell", () => {
    const none = fixture();
    none.synapses = null;
    expect(buildStories(none, live).map((s) => s.id)).toEqual(["axon"]);
  });
});

describe("StoryPlayer.cancel", () => {
  it("ends the story where the camera is, with no flight, and still takes the wiring down", () => {
    const bus = new EventBus();
    const log: string[] = [];
    bus.on("story", (e) => log.push(`story:${e.id}`));
    bus.on("partners", (e) => log.push(`partners:${e.show}`));
    let flights = 0;
    const player = new StoryPlayer({ bus, fly: () => flights++, place: () => {}, rest: () => ({ position: new Vector3(), target: new Vector3() }) });
    player.start({ id: "c", title: "C", blurb: "", neuron: 1, seconds: 30, steps: [{ at: 0, do: { kind: "partners", neuron: 1, show: true } }] });
    player.cancel();
    expect(player.running).toBe(false);
    expect(flights).toBe(0);
    expect(log).toEqual(["story:c", "partners:true", "partners:false", "story:null"]);
  });
});

describe("StoryPlayer speed", () => {
  it("slows the simulation at its cue and puts it back to normal when the story ends, however it ends", () => {
    for (const how of ["done", "skip", "cancel"] as const) {
      const bus = new EventBus();
      const scales: number[] = [];
      bus.on("timeScale", (e) => scales.push(e.scale));
      const player = new StoryPlayer({ bus, fly: () => {}, place: () => {}, rest: () => ({ position: new Vector3(), target: new Vector3() }) });
      player.start({ id: "s", title: "S", blurb: "", neuron: 1, seconds: 3, steps: [{ at: 1, do: { kind: "speed", scale: 0.2 } }] });
      for (let i = 0; i < 90; i++) player.update(1 / 60);
      expect(scales).toEqual([0.2]);
      if (how === "skip") player.skip();
      else if (how === "cancel") player.cancel();
      else for (let i = 0; i < 120; i++) player.update(1 / 60);
      expect(scales).toEqual([0.2, 1]);
    }
  });
});

describe("StoryPlayer live captions", () => {
  it("re-reads a live caption every time it is asked, so a count can climb on screen", () => {
    let fired = 1;
    const player = new StoryPlayer({ bus: new EventBus(), fly: () => {}, place: () => {}, rest: () => ({ position: new Vector3(), target: new Vector3() }) });
    player.start({ id: "l", title: "L", blurb: "", neuron: 1, seconds: 30, steps: [{ at: 0, do: { kind: "caption", text: () => `${fired} fired`, seconds: 5, live: true } }] });
    player.update(1 / 60);
    expect(player.caption()).toBe("1 fired");
    fired = 4;
    expect(player.caption()).toBe("4 fired");
  });
});
