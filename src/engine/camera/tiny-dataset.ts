/**
 * A hand-built two-neuron dataset for unit tests (no GPU, no fetch). Bounds are 0..1000 µm on every
 * axis, so quantised positions are within 0.02 µm of the values written here.
 *
 *   neuron 0 (excitatory "23P", soma 100,200,100)
 *     0 soma -> 1 axon (110,200,100) pd 10 -> 2 axon (130,200,100) pd 30 -> 3 axon (130,230,100) pd 60
 *     -> 4 axon (160,230,100) pd 90 (the farthest tip);  2 -> 5 axon (130,200,120) pd 50 (short branch)
 *     0 -> 6 dendrite (100,180,100) pd 20
 *   neuron 1 (inhibitory "BC", soma 300,230,100)
 *     7 soma -> 8 dendrite (200,230,100) pd 100 -> 9 dendrite (130,230,100) pd 170;  7 -> 10 axon (320,230,100) pd 20
 *   one synapse: neuron 0 -> neuron 1 at (130,230,100), 60 µm down neuron 0's axon.
 */

import { Compartment, NO_PARENT, PATH_DIST_UNIT_UM, quantise, type Manifest, type SynapseTable } from "../format";
import type { Dataset } from "../data";

const MAX = 1000;

interface Node {
  p: [number, number, number];
  parent: number;
  pd: number;
  comp: Compartment;
}

const NODES: Node[] = [
  { p: [100, 200, 100], parent: -1, pd: 0, comp: Compartment.Soma },
  { p: [110, 200, 100], parent: 0, pd: 10, comp: Compartment.Axon },
  { p: [130, 200, 100], parent: 1, pd: 30, comp: Compartment.Axon },
  { p: [130, 230, 100], parent: 2, pd: 60, comp: Compartment.Axon },
  { p: [160, 230, 100], parent: 3, pd: 90, comp: Compartment.Axon },
  { p: [130, 200, 120], parent: 2, pd: 50, comp: Compartment.Axon },
  { p: [100, 180, 100], parent: 0, pd: 20, comp: Compartment.Dendrite },
  { p: [300, 230, 100], parent: -1, pd: 0, comp: Compartment.Soma },
  { p: [200, 230, 100], parent: 7, pd: 100, comp: Compartment.Dendrite },
  { p: [130, 230, 100], parent: 8, pd: 170, comp: Compartment.Dendrite },
  { p: [320, 230, 100], parent: 7, pd: 20, comp: Compartment.Axon },
];

export function tinyDataset(options: { synapses?: boolean } = {}): Dataset {
  const withSynapses = options.synapses ?? true;
  const n = NODES.length;
  const pos = new Uint16Array(n * 3);
  NODES.forEach((node, i) => node.p.forEach((v, k) => (pos[i * 3 + k] = quantise(v, 0, MAX))));
  const manifest: Manifest = {
    version: 1,
    boundsUm: { min: [0, 0, 0], max: [MAX, MAX, MAX] },
    neuronCount: 2,
    cellTypes: ["23P", "BC"],
    neurons: "neurons.bin",
    lods: {
      hi: { nodeCount: n, chunks: [] },
      lite: { nodeCount: n, chunks: [] },
    },
    synapses: withSynapses ? { file: "synapses.bin", count: 1 } : null,
    credits: { dataset: "tiny", licence: "CC BY 4.0", url: "https://example.org", citations: ["A citation."] },
  };
  const synapses: SynapseTable | null = withSynapses
    ? {
        count: 1,
        neuronCount: 2,
        preOffsets: new Uint32Array([0, 1, 1]),
        pre: new Uint16Array([0]),
        post: new Uint16Array([1]),
        pos: new Uint16Array([quantise(130, 0, MAX), quantise(230, 0, MAX), quantise(100, 0, MAX)]),
        preDistQ: new Uint16Array([60 / PATH_DIST_UNIT_UM]),
        size: new Uint8Array([128]),
      }
    : null;
  return {
    manifest,
    lod: "hi",
    neurons: {
      count: 2,
      rootId: new BigUint64Array([864691135000000001n, 864691135000000002n]),
      somaUm: new Float32Array([100, 200, 100, 300, 230, 100]),
      cellType: new Uint8Array([0, 1]),
      inhibitory: new Uint8Array([0, 1]),
      layer: new Uint8Array([2, 2]),
    },
    nodes: {
      count: n,
      pos,
      radiusNm: new Uint16Array(n).fill(500),
      parent: Uint32Array.from(NODES.map((x) => (x.parent < 0 ? NO_PARENT : x.parent))),
      pathDistQ: Uint16Array.from(NODES.map((x) => x.pd / PATH_DIST_UNIT_UM)),
      compartment: Uint8Array.from(NODES.map((x) => x.comp)),
      neuronOfNode: Uint16Array.from(NODES.map((_, i) => (i < 7 ? 0 : 1))),
    },
    neuronNodeStart: new Uint32Array([0, 7]),
    neuronNodeCount: new Uint32Array([7, 4]),
    synapses,
  };
}
