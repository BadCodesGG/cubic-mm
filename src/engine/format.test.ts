import { describe, expect, it } from "vitest";
import {
  Compartment,
  NO_PARENT,
  decodeChunk,
  decodeNeurons,
  decodeSynapses,
  dequantise,
  encodeChunk,
  encodeNeurons,
  encodeSynapses,
  parseManifest,
  quantise,
  type Chunk,
  type NeuronTable,
  type SynapseTable,
} from "./format";

describe("quantisation", () => {
  it("round-trips within one step of the range", () => {
    const [min, max] = [110.592, 1814.528];
    for (const v of [min, 500.25, 1234.5, max]) {
      expect(Math.abs(dequantise(quantise(v, min, max), min, max) - v)).toBeLessThan((max - min) / 65535);
    }
  });
});

describe("neurons.bin", () => {
  it("round-trips through encode and decode", () => {
    const t: NeuronTable = {
      count: 3,
      rootId: new BigUint64Array([864691134884807418n, 1n, 2n ** 63n]),
      somaUm: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
      cellType: new Uint8Array([0, 1, 2]),
      inhibitory: new Uint8Array([0, 1, 0]),
      layer: new Uint8Array([2, 4, 5]),
    };
    const d = decodeNeurons(encodeNeurons(t));
    expect(d.count).toBe(3);
    expect([...d.rootId]).toEqual([...t.rootId]);
    expect([...d.somaUm]).toEqual([...t.somaUm]);
    expect([...d.cellType]).toEqual([0, 1, 2]);
    expect([...d.inhibitory]).toEqual([0, 1, 0]);
    expect([...d.layer]).toEqual([2, 4, 5]);
  });
});

describe("chunk", () => {
  it("round-trips an odd node count, which forces padding before the u8 tail", () => {
    const nodeCount = 5;
    const c: Chunk = {
      nodeStart: 100,
      nodeCount,
      neurons: [
        { neuronIndex: 7, nodeStart: 100, nodeCount: 3 },
        { neuronIndex: 9, nodeStart: 103, nodeCount: 2 },
      ],
      parent: new Uint32Array([NO_PARENT, 100, 101, NO_PARENT, 103]),
      pos: new Uint16Array(Array.from({ length: nodeCount * 3 }, (_, i) => i * 1000)),
      radiusNm: new Uint16Array([6000, 300, 134, 5000, 200]),
      pathDistQ: new Uint16Array([0, 16, 40, 0, 12]),
      compartment: new Uint8Array([Compartment.Soma, Compartment.Axon, Compartment.Axon, Compartment.Soma, Compartment.Dendrite]),
    };
    const buf = encodeChunk(c);
    expect(buf.byteLength % 4).toBe(0);
    const d = decodeChunk(buf);
    expect(d.nodeStart).toBe(100);
    expect(d.nodeCount).toBe(5);
    expect(d.neurons).toEqual(c.neurons);
    expect([...d.parent]).toEqual([...c.parent]);
    expect([...d.pos]).toEqual([...c.pos]);
    expect([...d.radiusNm]).toEqual([...c.radiusNm]);
    expect([...d.pathDistQ]).toEqual([...c.pathDistQ]);
    expect([...d.compartment]).toEqual([...c.compartment]);
  });

  it("rejects a file with the wrong magic", () => {
    expect(() => decodeChunk(new ArrayBuffer(16))).toThrow(/magic/i);
  });
});

describe("synapses.bin", () => {
  it("round-trips and keeps the pre-sorted offsets", () => {
    const t: SynapseTable = {
      count: 3,
      neuronCount: 2,
      preOffsets: new Uint32Array([0, 2, 3]),
      pre: new Uint16Array([0, 0, 1]),
      post: new Uint16Array([1, 1, 0]),
      pos: new Uint16Array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
      preDistQ: new Uint16Array([40, 800, 12]),
      size: new Uint8Array([10, 200, 5]),
    };
    const d = decodeSynapses(encodeSynapses(t));
    expect(d.count).toBe(3);
    expect(d.neuronCount).toBe(2);
    expect([...d.preOffsets]).toEqual([0, 2, 3]);
    expect([...d.pre]).toEqual([0, 0, 1]);
    expect([...d.post]).toEqual([1, 1, 0]);
    expect([...d.pos]).toEqual([...t.pos]);
    expect([...d.preDistQ]).toEqual([40, 800, 12]);
    expect([...d.size]).toEqual([10, 200, 5]);
  });
});

describe("manifest", () => {
  it("accepts the current version and rejects a missing LOD", () => {
    const good = {
      version: 1,
      boundsUm: { min: [0, 0, 0], max: [1, 1, 1] },
      neuronCount: 0,
      cellTypes: [],
      neurons: "neurons.bin",
      lods: { hi: { nodeCount: 0, chunks: [] }, lite: { nodeCount: 0, chunks: [] } },
      synapses: null,
      credits: { dataset: "", licence: "", url: "", citations: [] },
    };
    expect(parseManifest(good).version).toBe(1);
    expect(() => parseManifest({ ...good, lods: { hi: good.lods.hi } })).toThrow(/LOD/);
    expect(() => parseManifest({ ...good, version: 2 })).toThrow(/version/);
  });
});
