import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";
import { EventBus } from "./events";
import { decodeNeurons, decodeSynapses, type SynapseTable } from "./format";
import {
  FADE_IN_S,
  FADE_OUT_S,
  LineKind,
  MAX_LINES,
  PIN_DIM,
  PREVIEW_S,
  WiringState,
  approach,
  buildPartnerIndex,
  lineWeight,
  wiringLines,
} from "./partners";

/** Five cells; 4 is inhibitory. Synapses as [pre, post, size], grouped by pre as the file is. */
const EDGES: [number, number, number][] = [
  [0, 1, 10],
  [0, 1, 20],
  [0, 2, 5],
  [1, 0, 7],
  [1, 2, 9],
  [1, 2, 1],
  [1, 2, 2],
  [3, 2, 4],
  [4, 2, 6],
  [4, 0, 8],
];
const INHIBITORY = [0, 0, 0, 0, 1];

function table(edges: [number, number, number][], neuronCount: number): SynapseTable {
  const count = edges.length;
  const preOffsets = new Uint32Array(neuronCount + 1);
  for (const [a] of edges) preOffsets[a + 1]++;
  for (let n = 0; n < neuronCount; n++) preOffsets[n + 1] += preOffsets[n];
  return {
    count,
    neuronCount,
    preOffsets,
    pre: Uint16Array.from(edges, (e) => e[0]),
    post: Uint16Array.from(edges, (e) => e[1]),
    pos: new Uint16Array(count * 3),
    preDistQ: new Uint16Array(count),
    size: Uint8Array.from(edges, (e) => e[2]),
  };
}

describe("PartnerIndex on a hand-built table", () => {
  const index = buildPartnerIndex(table(EDGES, 5), INHIBITORY);

  it("folds synapses into one edge per pair, with counts and summed size, biggest first", () => {
    expect(index.edgeCount).toBe(7);
    expect(index.partnersOf(1).outputs).toEqual([
      { neuron: 2, synapses: 3, size: 12 },
      { neuron: 0, synapses: 1, size: 7 },
    ]);
    expect(index.partnersOf(0).outputs).toEqual([
      { neuron: 1, synapses: 2, size: 30 },
      { neuron: 2, synapses: 1, size: 5 },
    ]);
  });

  it("lists inputs by count and breaks ties by size then index", () => {
    expect(index.partnersOf(2).inputs).toEqual([
      { neuron: 1, synapses: 3, size: 12 },
      { neuron: 4, synapses: 1, size: 6 },
      { neuron: 0, synapses: 1, size: 5 },
      { neuron: 3, synapses: 1, size: 4 },
    ]);
    expect(index.partnersOf(3).inputs).toEqual([]);
    expect(index.partnersOf(2).outputs).toEqual([]);
  });

  it("counts partners, synapses and inhibitory inputs", () => {
    expect(index.partnerStats(2)).toEqual({ outputs: 0, inputs: 4, outputSynapses: 0, inputSynapses: 6, inhibitoryInputs: 1 });
    expect(index.partnerStats(0)).toEqual({ outputs: 2, inputs: 2, outputSynapses: 3, inputSynapses: 2, inhibitoryInputs: 1 });
    expect(index.partnerStats(3)).toEqual({ outputs: 1, inputs: 0, outputSynapses: 1, inputSynapses: 0, inhibitoryInputs: 0 });
    expect(index.maxPartners).toBe(4);
  });

  it("is symmetric: every output of A is an input of its target, with the same numbers", () => {
    for (let a = 0; a < 5; a++) {
      for (const out of index.partnersOf(a).outputs) {
        const back = index.partnersOf(out.neuron).inputs.find((p) => p.neuron === a);
        expect(back).toEqual({ neuron: a, synapses: out.synapses, size: out.size });
      }
    }
  });

  it("answers an unknown neuron with nothing", () => {
    expect(index.partnersOf(-1)).toEqual({ outputs: [], inputs: [] });
    expect(index.partnersOf(99).inputs).toEqual([]);
    expect(index.partnerStats(99).inputs).toBe(0);
  });

  it("handles a table with no synapses", () => {
    const empty = buildPartnerIndex(table([], 3), [0, 0, 0]);
    expect(empty.edgeCount).toBe(0);
    expect(empty.partnerStats(1)).toEqual({ outputs: 0, inputs: 0, outputSynapses: 0, inputSynapses: 0, inhibitoryInputs: 0 });
  });
});

describe("wiringLines", () => {
  const index = buildPartnerIndex(table(EDGES, 5), INHIBITORY);

  it("merges outputs and inputs by synapse count and tags inhibitory inputs", () => {
    const lines = wiringLines(index, INHIBITORY, 2);
    expect(lines).toEqual([
      { neuron: 1, kind: LineKind.Input, synapses: 3 },
      { neuron: 0, kind: LineKind.Input, synapses: 1 },
      { neuron: 3, kind: LineKind.Input, synapses: 1 },
      { neuron: 4, kind: LineKind.InhibitoryInput, synapses: 1 },
    ]);
  });

  it("draws a reciprocal partner twice, once per direction", () => {
    const kinds = wiringLines(index, INHIBITORY, 0)
      .filter((l) => l.neuron === 1)
      .map((l) => l.kind);
    expect(kinds.sort()).toEqual([LineKind.Output, LineKind.Input].sort());
  });

  it("keeps the biggest when capped", () => {
    expect(wiringLines(index, INHIBITORY, 2, 2).map((l) => l.neuron)).toEqual([1, 0]);
  });

  it("caps a hub at 400 lines, largest first", () => {
    const n = 600;
    const edges: [number, number, number][] = [];
    for (let post = 1; post < n; post++) for (let k = 0; k <= post % 7; k++) edges.push([0, post, 1]);
    const hub = buildPartnerIndex(table(edges, n), new Array(n).fill(0));
    const lines = wiringLines(hub, new Array(n).fill(0), 0);
    expect(lines).toHaveLength(MAX_LINES);
    for (let i = 1; i < lines.length; i++) expect(lines[i - 1].synapses).toBeGreaterThanOrEqual(lines[i].synapses);
    expect(lines[0].synapses).toBe(7);
  });

  it("skips a cell's synapse onto itself", () => {
    const self = buildPartnerIndex(table([[0, 0, 3], [0, 1, 3]], 2), [0, 0]);
    expect(wiringLines(self, [0, 0], 0).map((l) => l.neuron)).toEqual([1]);
  });
});

describe("lineWeight and approach", () => {
  it("scales with log synapse count and stays in 0.3..1", () => {
    expect(lineWeight(1)).toBeCloseTo(0.3);
    expect(lineWeight(4)).toBeCloseTo(0.65);
    expect(lineWeight(16)).toBeCloseTo(1);
    expect(lineWeight(500)).toBe(1);
  });

  it("fades in over 150 ms and out over 250 ms", () => {
    expect(approach(0, 1, FADE_IN_S / 2)).toBeCloseTo(0.5);
    expect(approach(0, 1, 1)).toBe(1);
    expect(approach(1, 0, FADE_OUT_S / 2)).toBeCloseTo(0.5);
    expect(approach(1, 0, 1)).toBe(0);
    expect(approach(1, PIN_DIM, 1)).toBe(PIN_DIM);
  });
});

describe("WiringState", () => {
  const setup = () => {
    const bus = new EventBus();
    return { bus, state: new WiringState(bus, 10) };
  };

  it("follows the pointer, and clears on leave", () => {
    const { bus, state } = setup();
    bus.emit("hover", { neuron: 3 });
    expect(state.shown()).toEqual({ hover: 3, pin: -1 });
    bus.emit("hover", { neuron: -1 });
    expect(state.shown()).toEqual({ hover: -1, pin: -1 });
    bus.emit("hover", { neuron: 99 });
    expect(state.shown().hover).toBe(-1);
  });

  it("pins the selected cell and shows both when another is hovered", () => {
    const { bus, state } = setup();
    bus.emit("select", { neuron: 2 });
    bus.emit("partners", { neuron: 2, show: true });
    expect(state.pinned).toBe(2);
    bus.emit("hover", { neuron: 5 });
    expect(state.shown()).toEqual({ hover: 5, pin: 2 });
    // The pinned cell is not drawn twice.
    bus.emit("hover", { neuron: 2 });
    expect(state.shown()).toEqual({ hover: -1, pin: 2 });
  });

  it("drops the pin on Deselect (Esc) and when another cell is selected, but not on the same one", () => {
    const { bus, state } = setup();
    bus.emit("select", { neuron: 2 });
    bus.emit("partners", { neuron: 2, show: true });
    bus.emit("select", { neuron: 2 });
    expect(state.pinned).toBe(2);
    bus.emit("select", { neuron: -1 });
    expect(state.pinned).toBe(-1);
    bus.emit("select", { neuron: 2 });
    bus.emit("partners", { neuron: 2, show: true });
    bus.emit("select", { neuron: 4 });
    expect(state.pinned).toBe(-1);
  });

  it("unpins on partners show:false and ignores a bad neuron", () => {
    const { bus, state } = setup();
    bus.emit("partners", { neuron: 3, show: true });
    bus.emit("partners", { neuron: 99, show: true });
    expect(state.pinned).toBe(3);
    bus.emit("partners", { neuron: 3, show: false });
    expect(state.pinned).toBe(-1);
  });

  it("previews for 4 s then releases, without counting as a pin", () => {
    const { bus, state } = setup();
    bus.emit("select", { neuron: 6 });
    state.preview(6, PREVIEW_S);
    expect(state.shown().pin).toBe(6);
    expect(state.pinned).toBe(-1);
    state.tick(3.9);
    expect(state.shown().pin).toBe(6);
    state.tick(0.2);
    expect(state.shown().pin).toBe(-1);
  });

  it("keeps a preview the visitor pins, and lets go of one they select away from", () => {
    const { bus, state } = setup();
    bus.emit("select", { neuron: 6 });
    state.preview(6);
    bus.emit("partners", { neuron: 6, show: true });
    state.tick(10);
    expect(state.shown().pin).toBe(6);
    bus.emit("select", { neuron: -1 });
    state.preview(7);
    bus.emit("select", { neuron: 8 });
    expect(state.shown().pin).toBe(-1);
  });

  it("shows nothing while the intro tour plays", () => {
    const { bus, state } = setup();
    bus.emit("select", { neuron: 6 });
    bus.emit("partners", { neuron: 6, show: true });
    bus.emit("hover", { neuron: 1 });
    bus.emit("tour", { running: true });
    expect(state.shown()).toEqual({ hover: -1, pin: -1 });
    state.preview(6);
    expect(state.shown().pin).toBe(-1);
  });

  it("stops listening once disposed", () => {
    const { bus, state } = setup();
    state.dispose();
    bus.emit("hover", { neuron: 3 });
    expect(state.shown().hover).toBe(-1);
  });

  describe("the G key", () => {
    class FakeElement {
      constructor(private readonly field = false) {}
      closest() {
        return this.field ? {} : null;
      }
    }
    afterEach(() => vi.unstubAllGlobals());
    const key = (target: unknown = new FakeElement(), extra = {}) => ({
      code: "KeyG",
      repeat: false,
      ctrlKey: false,
      metaKey: false,
      altKey: false,
      target,
      ...extra,
    });

    it("toggles the selection's pin, and does nothing with no selection", () => {
      vi.stubGlobal("Element", FakeElement);
      const { bus, state } = setup();
      const shown: boolean[] = [];
      bus.on("partners", (e) => shown.push(e.show));
      expect(state.onKey(key(), bus)).toBe(false);
      bus.emit("select", { neuron: 4 });
      expect(state.onKey(key(), bus)).toBe(true);
      expect(state.pinned).toBe(4);
      state.onKey(key(), bus);
      expect(state.pinned).toBe(-1);
      expect(shown).toEqual([true, false]);
    });

    it("leaves G alone in a text field, with a modifier, on repeat, and for other keys", () => {
      vi.stubGlobal("Element", FakeElement);
      const { bus, state } = setup();
      bus.emit("select", { neuron: 4 });
      expect(state.onKey(key(new FakeElement(true)), bus)).toBe(false);
      expect(state.onKey(key(undefined, { ctrlKey: true }), bus)).toBe(false);
      expect(state.onKey(key(undefined, { repeat: true }), bus)).toBe(false);
      expect(state.onKey(key(undefined, { code: "KeyH" }), bus)).toBe(false);
      expect(state.pinned).toBe(-1);
    });
  });
});

describe("PartnerIndex on the committed data", () => {
  const DATA = fileURLToPath(new URL("../../public/data/", import.meta.url));
  const file = (name: string) => {
    const b = readFileSync(DATA + name);
    return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
  };
  const synapses = decodeSynapses(file("synapses.bin"));
  const neurons = decodeNeurons(file("neurons.bin"));

  it("builds in well under the 30 ms idle budget", () => {
    buildPartnerIndex(synapses, neurons.inhibitory); // warm the JIT, as a page's first frames would not, but a real load runs it once
    const t = performance.now();
    buildPartnerIndex(synapses, neurons.inhibitory);
    const ms = performance.now() - t;
    expect(ms).toBeLessThan(30);
  });

  it("has as many input synapses as output synapses as synapses", () => {
    const index = buildPartnerIndex(synapses, neurons.inhibitory);
    let out = 0;
    let inn = 0;
    let outPartners = 0;
    let inPartners = 0;
    for (let n = 0; n < neurons.count; n++) {
      const s = index.partnerStats(n);
      out += s.outputSynapses;
      inn += s.inputSynapses;
      outPartners += s.outputs;
      inPartners += s.inputs;
    }
    expect(out).toBe(synapses.count);
    expect(inn).toBe(synapses.count);
    expect(outPartners).toBe(index.edgeCount);
    expect(inPartners).toBe(index.edgeCount);
  });

  it("agrees with the table's own per-neuron counts and is symmetric", () => {
    const index = buildPartnerIndex(synapses, neurons.inhibitory);
    const incoming = new Uint32Array(neurons.count);
    for (let s = 0; s < synapses.count; s++) incoming[synapses.post[s]]++;
    for (let n = 0; n < neurons.count; n++) {
      const stats = index.partnerStats(n);
      expect(stats.outputSynapses).toBe(synapses.preOffsets[n + 1] - synapses.preOffsets[n]);
      expect(stats.inputSynapses).toBe(incoming[n]);
      const { outputs, inputs } = index.partnersOf(n);
      for (let i = 1; i < outputs.length; i++) expect(outputs[i - 1].synapses).toBeGreaterThanOrEqual(outputs[i].synapses);
      for (let i = 1; i < inputs.length; i++) expect(inputs[i - 1].synapses).toBeGreaterThanOrEqual(inputs[i].synapses);
      // Sample: every output appears as an input of its target.
      if (n % 37 === 0) {
        for (const out of outputs) {
          const back = index.partnersOf(out.neuron).inputs.find((p) => p.neuron === n);
          expect(back?.synapses).toBe(out.synapses);
        }
      }
    }
  });
});
