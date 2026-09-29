/**
 * A stand-in synapse table for datasets that have none yet (the synthetic stand-in, and the real
 * data until the synapse pass lands): Peters' rule, which says axons synapse roughly wherever
 * they pass close to a dendrite.
 *
 * For every axon node, the nearest dendrite node of a different neuron within `radiusUm` becomes
 * a synapse at that dendrite node. Each connected pair keeps only its first contact (nearest the
 * presynaptic soma along the axon): an axon running alongside a dendrite passes within 3 µm of
 * it dozens of times, and those contacts would all fire together, enough to drive the post cell
 * on their own and tip the whole volume into runaway excitation. A presynaptic neuron keeps at
 * most `capPerPre` synapses. `size` is random. The result is sorted by pre, like the packed table.
 *
 * It is a fallback, not a connectome: it over-connects cells whose arbours merely overlap, and
 * the `mode` event carries `syntheticSynapses: true` whenever it is in use.
 */

import { dequantise, type SynapseTable } from "../format";
import type { Dataset } from "../data";

export interface PetersOptions {
  radiusUm?: number;
  capPerPre?: number;
}

/** `Compartment` values; a const enum does not survive isolated-module bundling across files. */
const AXON = 1;
const DENDRITE = 2;

type PetersInput = Pick<Dataset, "manifest" | "neurons" | "nodes" | "neuronNodeStart" | "neuronNodeCount">;

export function petersSynapses(
  data: PetersInput,
  random: () => number,
  { radiusUm = 3, capPerPre = 300 }: PetersOptions = {},
): SynapseTable {
  const { nodes, manifest } = data;
  const { min, max } = manifest.boundsUm;
  const neuronCount = data.neurons.count;
  const x = (i: number) => dequantise(nodes.pos[i * 3], min[0], max[0]);
  const y = (i: number) => dequantise(nodes.pos[i * 3 + 1], min[1], max[1]);
  const z = (i: number) => dequantise(nodes.pos[i * 3 + 2], min[2], max[2]);

  // Hash every dendrite node into cells of `radiusUm`, so a query only visits the 27 cells around it.
  const nx = Math.ceil((max[0] - min[0]) / radiusUm) + 1;
  const ny = Math.ceil((max[1] - min[1]) / radiusUm) + 1;
  const cellOf = (px: number, py: number, pz: number) =>
    Math.floor((px - min[0]) / radiusUm) + nx * (Math.floor((py - min[1]) / radiusUm) + ny * Math.floor((pz - min[2]) / radiusUm));
  const dendrites: number[] = [];
  for (let i = 0; i < nodes.count; i++) if (nodes.compartment[i] === DENDRITE) dendrites.push(i);
  const keys = new Float64Array(dendrites.length);
  dendrites.forEach((i, k) => (keys[k] = cellOf(x(i), y(i), z(i))));
  const order = Array.from(dendrites.keys()).sort((a, b) => keys[a] - keys[b]);
  const sorted = Int32Array.from(order, (k) => dendrites[k]);
  const cells = new Map<number, [number, number]>();
  for (let k = 0; k < order.length; k++) {
    const key = keys[order[k]];
    const hit = cells.get(key);
    if (hit) hit[1] = k + 1;
    else cells.set(key, [k, k + 1]);
  }

  const pre: number[] = [];
  const post: number[] = [];
  const at: number[] = [];
  const preDistQ: number[] = [];
  const preOffsets = new Uint32Array(neuronCount + 1);
  const r2 = radiusUm * radiusUm;
  for (let n = 0; n < neuronCount; n++) {
    preOffsets[n] = pre.length;
    /** Post neurons this neuron already contacts. */
    const used = new Set<number>();
    const start = data.neuronNodeStart[n];
    const end = start + data.neuronNodeCount[n];
    for (let i = start; i < end && used.size < capPerPre; i++) {
      if (nodes.compartment[i] !== AXON) continue;
      const px = x(i);
      const py = y(i);
      const pz = z(i);
      const cx = Math.floor((px - min[0]) / radiusUm);
      const cy = Math.floor((py - min[1]) / radiusUm);
      const cz = Math.floor((pz - min[2]) / radiusUm);
      let best = -1;
      let bestD = r2;
      for (let dz = -1; dz <= 1; dz++) {
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const range = cells.get(cx + dx + nx * (cy + dy + ny * (cz + dz)));
            if (!range) continue;
            for (let k = range[0]; k < range[1]; k++) {
              const j = sorted[k];
              if (nodes.neuronOfNode[j] === n) continue;
              const ex = x(j) - px;
              const ey = y(j) - py;
              const ez = z(j) - pz;
              const d = ex * ex + ey * ey + ez * ez;
              if (d <= bestD) {
                bestD = d;
                best = j;
              }
            }
          }
        }
      }
      if (best < 0 || used.has(nodes.neuronOfNode[best])) continue;
      used.add(nodes.neuronOfNode[best]);
      pre.push(n);
      post.push(nodes.neuronOfNode[best]);
      at.push(best);
      preDistQ.push(nodes.pathDistQ[i]);
    }
  }
  preOffsets[neuronCount] = pre.length;

  const count = pre.length;
  const pos = new Uint16Array(count * 3);
  const size = new Uint8Array(count);
  for (let s = 0; s < count; s++) {
    pos.set(nodes.pos.subarray(at[s] * 3, at[s] * 3 + 3), s * 3);
    size[s] = Math.min(255, Math.floor(random() * 256));
  }
  return {
    count,
    neuronCount,
    preOffsets,
    pre: Uint16Array.from(pre),
    post: Uint16Array.from(post),
    pos,
    preDistQ: Uint16Array.from(preDistQ),
    size,
  };
}
