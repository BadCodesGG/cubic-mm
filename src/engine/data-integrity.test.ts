import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  Compartment,
  NO_PARENT,
  decodeChunk,
  decodeNeurons,
  decodeSynapses,
  dequantise,
  parseManifest,
  type Chunk,
  type LodName,
} from "./format";

/**
 * Structural checks on the committed `public/data`, so a bad `pipeline/pack.py` run fails here
 * rather than as a corrupt scene. Positions are uint16 by construction, so the bounds check is
 * done on the dequantised values against `manifest.boundsUm`.
 */

const DATA = fileURLToPath(new URL("../../public/data/", import.meta.url));

function read(rel: string): ArrayBuffer {
  const b = readFileSync(DATA + rel);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
}

const manifest = parseManifest(JSON.parse(readFileSync(DATA + "manifest.json", "utf8")));
const neurons = decodeNeurons(read(manifest.neurons));
const { min, max } = manifest.boundsUm;

/** Per-LOD edge length caps from `LODS` in the packer, plus room for two quantisation half-steps per end. */
const MAX_EDGE_UM: Record<LodName, number> = { hi: 28 + 0.25, lite: 150 + 0.25 };

/** The node budgets the packer enforces for the full set. */
const NODE_BUDGET: Record<LodName, number> = { hi: 1_200_000, lite: 350_000 };

function dequantisedPoint(chunk: Chunk, local: number): number[] {
  return [0, 1, 2].map((k) => dequantise(chunk.pos[local * 3 + k], min[k], max[k]));
}

describe("public/data", () => {
  it("has a neuron table that agrees with the manifest", () => {
    expect(neurons.count).toBe(manifest.neuronCount);
    for (let i = 0; i < neurons.count; i++) {
      expect(neurons.cellType[i]).toBeLessThan(manifest.cellTypes.length);
      expect(neurons.layer[i]).toBeGreaterThanOrEqual(0);
      expect(neurons.layer[i]).toBeLessThanOrEqual(6);
      expect([0, 1]).toContain(neurons.inhibitory[i]);
    }
    expect(new Set(neurons.rootId).size).toBe(neurons.count);
  });

  for (const lod of ["hi", "lite"] as LodName[]) {
    describe(lod, () => {
      const info = manifest.lods[lod];
      const chunks = info.chunks.map((c) => decodeChunk(read(c.file)));

      it("has chunk headers that match the manifest, contiguously", () => {
        expect(info.nodeCount).toBe(info.chunks.reduce((s, c) => s + c.nodeCount, 0));
        let next = 0;
        info.chunks.forEach((c, i) => {
          const chunk = chunks[i];
          expect(c.nodeStart).toBe(next);
          expect(chunk.nodeStart).toBe(c.nodeStart);
          expect(chunk.nodeCount).toBe(c.nodeCount);
          expect(chunk.neurons.length).toBe(c.neuronCount);
          expect(chunk.parent.length).toBe(c.nodeCount);
          expect(chunk.pos.length).toBe(c.nodeCount * 3);
          let at = c.nodeStart;
          for (const n of chunk.neurons) {
            expect(n.nodeStart).toBe(at);
            at += n.nodeCount;
          }
          expect(at).toBe(c.nodeStart + c.nodeCount);
          next += c.nodeCount;
        });
        expect(chunks.reduce((s, c) => s + c.neurons.length, 0)).toBe(manifest.neuronCount);
      });

      it("stays under the node budget and gives every neuron at least one node", () => {
        if (manifest.neuronCount > 1000) expect(info.nodeCount).toBeLessThanOrEqual(NODE_BUDGET[lod]);
        for (const chunk of chunks) {
          for (const n of chunk.neurons) expect(n.nodeCount).toBeGreaterThanOrEqual(1);
        }
      });

      it("keeps every neuron a soma-rooted tree with parents first and pathDist rising", () => {
        const seen = new Set<number>();
        for (const chunk of chunks) {
          for (const n of chunk.neurons) {
            seen.add(n.neuronIndex);
            const first = n.nodeStart - chunk.nodeStart;
            expect(chunk.compartment[first]).toBe(Compartment.Soma);
            expect(chunk.parent[first]).toBe(NO_PARENT);
            expect(chunk.pathDistQ[first]).toBe(0);
            expect(n.nodeCount).toBeLessThan(65535);

            for (let g = n.nodeStart + 1; g < n.nodeStart + n.nodeCount; g++) {
              const local = g - chunk.nodeStart;
              const parent = chunk.parent[local];
              if (parent < n.nodeStart || parent >= g) {
                throw new Error(`${lod}: neuron ${n.neuronIndex} node ${g} has parent ${parent} outside [${n.nodeStart}, ${g})`);
              }
              if (chunk.pathDistQ[local] < chunk.pathDistQ[parent - chunk.nodeStart]) {
                throw new Error(`${lod}: neuron ${n.neuronIndex} node ${g} pathDist falls below its parent's`);
              }
            }
          }
        }
        expect(seen.size).toBe(manifest.neuronCount);
      });

      it(`places nodes inside the bounds, the soma on the table, and no edge over ${MAX_EDGE_UM[lod] - 0.25} um`, () => {
        info.chunks.forEach((c, i) => {
          const chunk = chunks[i];
          for (const n of chunk.neurons) {
            const soma = dequantisedPoint(chunk, n.nodeStart - chunk.nodeStart);
            for (let k = 0; k < 3; k++) {
              expect(Math.abs(soma[k] - neurons.somaUm[n.neuronIndex * 3 + k])).toBeLessThan(2);
            }
            for (let g = n.nodeStart + 1; g < n.nodeStart + n.nodeCount; g++) {
              const a = dequantisedPoint(chunk, g - chunk.nodeStart);
              const b = dequantisedPoint(chunk, chunk.parent[g - chunk.nodeStart] - chunk.nodeStart);
              if (Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) > MAX_EDGE_UM[lod]) {
                throw new Error(`${lod}: neuron ${n.neuronIndex} node ${g} is more than ${MAX_EDGE_UM[lod]} um from its parent`);
              }
            }
          }
          for (let local = 0; local < chunk.nodeCount; local++) {
            const p = dequantisedPoint(chunk, local);
            for (let k = 0; k < 3; k++) {
              if (p[k] < c.bboxUm.min[k] - 1e-9 || p[k] > c.bboxUm.max[k] + 1e-9) {
                throw new Error(`${c.file}: node ${local} is outside its manifest bbox`);
              }
            }
          }
        });
        for (let k = 0; k < 3; k++) {
          for (const c of info.chunks) {
            expect(c.bboxUm.min[k]).toBeGreaterThanOrEqual(min[k] - 1e-6);
            expect(c.bboxUm.max[k]).toBeLessThanOrEqual(max[k] + 1e-6);
          }
        }
      });
    });
  }

  describe("synapses", () => {
    const info = manifest.synapses;
    const table = info ? decodeSynapses(read(info.file)) : null;
    const hiChunks = manifest.lods.hi.chunks.map((c) => decodeChunk(read(c.file)));

    /** Largest pathDistQ of each neuron in the hi LOD. */
    const maxPathQ = new Uint16Array(manifest.neuronCount);
    for (const chunk of hiChunks) {
      for (const n of chunk.neurons) {
        for (let g = n.nodeStart; g < n.nodeStart + n.nodeCount; g++) {
          maxPathQ[n.neuronIndex] = Math.max(maxPathQ[n.neuronIndex], chunk.pathDistQ[g - chunk.nodeStart]);
        }
      }
    }

    it("is listed in the manifest and agrees with its header", () => {
      expect(info).not.toBeNull();
      expect(table!.count).toBe(info!.count);
      expect(table!.neuronCount).toBe(manifest.neuronCount);
      expect(table!.count).toBeGreaterThan(0);
    });

    it("points every synapse at real neurons", () => {
      for (let i = 0; i < table!.count; i++) {
        if (table!.pre[i] >= neurons.count || table!.post[i] >= neurons.count) {
          throw new Error(`synapse ${i} references neuron ${table!.pre[i]} -> ${table!.post[i]} of ${neurons.count}`);
        }
      }
    });

    it("is sorted by pre with preOffsets indexing the runs", () => {
      const { preOffsets, pre, count } = table!;
      expect(preOffsets.length).toBe(manifest.neuronCount + 1);
      expect(preOffsets[0]).toBe(0);
      expect(preOffsets[manifest.neuronCount]).toBe(count);
      for (let n = 0; n < manifest.neuronCount; n++) {
        const start = preOffsets[n];
        const end = preOffsets[n + 1];
        if (end < start) throw new Error(`preOffsets falls at neuron ${n}`);
        for (let i = start; i < end; i++) {
          if (pre[i] !== n) throw new Error(`synapse ${i} has pre ${pre[i]}, inside neuron ${n}'s run`);
        }
      }
      // Runs cover every synapse exactly once, so no synapse of pre n sits outside its run.
      let runTotal = 0;
      for (let n = 0; n < manifest.neuronCount; n++) runTotal += preOffsets[n + 1] - preOffsets[n];
      expect(runTotal).toBe(count);
    });

    it("keeps positions in bounds and preDistQ within the neuron's hi path length", () => {
      const { pre, pos, preDistQ, count } = table!;
      for (let i = 0; i < count; i++) {
        for (let k = 0; k < 3; k++) {
          const v = dequantise(pos[i * 3 + k], min[k], max[k]);
          if (v < min[k] - 1e-9 || v > max[k] + 1e-9) throw new Error(`synapse ${i} is outside the bounds`);
        }
        if (preDistQ[i] > maxPathQ[pre[i]] + 1) {
          throw new Error(`synapse ${i}: preDistQ ${preDistQ[i]} exceeds neuron ${pre[i]}'s longest path ${maxPathQ[pre[i]]}`);
        }
      }
    });

    it("uses the whole size scale, largest bucket 255", () => {
      const { size, count } = table!;
      let biggest = 0;
      for (let i = 0; i < count; i++) biggest = Math.max(biggest, size[i]);
      expect(biggest).toBe(255);
    });
  });

  it("groups the same neurons into the same chunks in both LODs", () => {
    const groups = (lod: LodName) =>
      manifest.lods[lod].chunks.map((c) => decodeChunk(read(c.file)).neurons.map((n) => n.neuronIndex));
    expect(groups("lite")).toEqual(groups("hi"));
  });
});
