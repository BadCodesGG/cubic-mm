import { describe, expect, it } from "vitest";
import { NO_PARENT, quantise, type Manifest } from "../format";
import type { Dataset } from "../data";
import { synthDataset } from "../synth";
import { petersSynapses } from "./peters";

const BOX = { min: [0, 0, 0] as [number, number, number], max: [100, 100, 100] as [number, number, number] };

/** Nodes as [neuron, compartment, x, y, z, pathUm]; each neuron's nodes must be contiguous, soma first. */
function tinyDataset(rows: [number, number, number, number, number, number][], neuronCount: number): Dataset {
  const count = rows.length;
  const nodes = {
    count,
    pos: new Uint16Array(count * 3),
    radiusNm: new Uint16Array(count),
    parent: new Uint32Array(count).fill(NO_PARENT),
    pathDistQ: new Uint16Array(count),
    compartment: new Uint8Array(count),
    neuronOfNode: new Uint16Array(count),
  };
  const neuronNodeStart = new Uint32Array(neuronCount);
  const neuronNodeCount = new Uint32Array(neuronCount);
  rows.forEach(([n, comp, x, y, z, path], i) => {
    if (neuronNodeCount[n] === 0) neuronNodeStart[n] = i;
    neuronNodeCount[n]++;
    nodes.pos.set([quantise(x, 0, 100), quantise(y, 0, 100), quantise(z, 0, 100)], i * 3);
    nodes.pathDistQ[i] = path * 4;
    nodes.compartment[i] = comp;
    nodes.neuronOfNode[i] = n;
  });
  return {
    manifest: { boundsUm: BOX } as Manifest,
    lod: "hi",
    neurons: {
      count: neuronCount,
      rootId: new BigUint64Array(neuronCount),
      somaUm: new Float32Array(neuronCount * 3),
      cellType: new Uint8Array(neuronCount),
      inhibitory: new Uint8Array(neuronCount),
      layer: new Uint8Array(neuronCount),
    },
    nodes,
    neuronNodeStart,
    neuronNodeCount,
    synapses: null,
  };
}

describe("petersSynapses (the fallback wiring when the dataset has no synapse table)", () => {
  it("connects an axon node to the nearest dendrite node of another neuron within 3 µm", () => {
    const data = tinyDataset(
      [
        [0, 0, 50, 50, 50, 0],
        [0, 1, 10, 10, 10, 40], // axon of 0
        [0, 2, 10, 11, 10, 5], // 0's own dendrite, 1 µm away: never a synapse
        [1, 0, 80, 80, 80, 0],
        [1, 2, 10, 12.5, 10, 30], // 2.5 µm away: the nearest of another neuron
        [1, 2, 12, 10, 10, 31], // 2 µm away: nearer still
        [1, 2, 10, 10, 14, 32], // 4 µm away: out of range
        [1, 1, 10, 10, 11, 40], // 1's axon, 1 µm away: not a dendrite, but it does reach 0's dendrite
      ],
      2,
    );
    const t = petersSynapses(data, () => 0.5);
    expect(t.count).toBe(2);
    expect(Array.from(t.preOffsets)).toEqual([0, 1, 2]);
    expect(t.pre[0]).toBe(0);
    expect(t.post[0]).toBe(1);
    expect(t.preDistQ[0]).toBe(160);
    expect(Array.from(t.pos.subarray(0, 3))).toEqual([quantise(12, 0, 100), quantise(10, 0, 100), quantise(10, 0, 100)]);
    expect(t.size[0]).toBe(128);
    // And the other way: 1's axon onto 0's dendrite, 1.4 µm away.
    expect([t.pre[1], t.post[1], t.preDistQ[1]]).toEqual([1, 0, 160]);
  });

  it("gives the synthetic dataset a synapse table sorted by pre, one synapse per connected pair, no self-synapses, at most 300 per neuron", () => {
    const data = synthDataset(1, 40);
    const t = data.synapses!;
    expect(t).not.toBeNull();
    expect(t.count).toBeGreaterThan(40);
    expect(t.neuronCount).toBe(40);
    for (let n = 0; n < 40; n++) {
      expect(t.preOffsets[n + 1] - t.preOffsets[n]).toBeLessThanOrEqual(300);
      for (let s = t.preOffsets[n]; s < t.preOffsets[n + 1]; s++) {
        expect(t.pre[s]).toBe(n);
        expect(t.post[s]).not.toBe(n);
      }
      // One contact per pair: an axon running alongside a dendrite would otherwise make a dozen
      // synapses that all fire together, enough to drive the post cell on their own.
      const posts = Array.from(t.post.subarray(t.preOffsets[n], t.preOffsets[n + 1]));
      expect(new Set(posts).size).toBe(posts.length);
    }
    expect(t.preOffsets[40]).toBe(t.count);
  });
});
