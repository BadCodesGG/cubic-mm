/**
 * What the HUD says about a neuron and about the whole volume. Pure functions over a `Dataset`:
 * every number is measured from the loaded arrays, nothing is typed in.
 *
 * Lengths are summed over whichever LOD the dataset was loaded at (`dataset.lod`). The hi LOD keeps
 * about every skeleton vertex, so its edge lengths are the cable length; the lite LOD is a
 * simplified skeleton and reads a little short.
 */

import { Compartment, NO_PARENT, dequantise } from "./format";
import type { Dataset } from "./data";
import { layerBoundaries } from "./scene/layers";

/** MICrONS minnie65 cell-type codes (aibs_cell_info `cell_type`) and what they mean. */
export const CELL_TYPE_NAMES: Readonly<Record<string, string>> = {
  "23P": "layer 2/3 pyramidal",
  "4P": "layer 4 pyramidal",
  "5P-IT": "layer 5 IT pyramidal",
  "5P-ET": "layer 5 ET pyramidal",
  "5P-NP": "layer 5 near-projecting pyramidal",
  "6P-IT": "layer 6 IT pyramidal",
  "6P-U": "layer 6 pyramidal, unclassified",
  "6P-CT": "layer 6 corticothalamic pyramidal",
  BC: "basket cell",
  MC: "Martinotti cell",
  BPC: "bipolar cell",
  NGC: "neurogliaform cell",
  "": "unclassified",
};

export function cellTypeName(code: string): string {
  return CELL_TYPE_NAMES[code] ?? code;
}

export interface NeuronInfo {
  index: number;
  typeCode: string;
  typeName: string;
  className: "excitatory" | "inhibitory";
  /** Cortical layer 1 to 6 by soma depth, 0 if unknown. */
  layer: number;
  layerName: string;
  /** MICrONS proofread root id, as a decimal string (it does not fit a JS number). */
  rootId: string;
  nodeCount: number;
  cableLengthUm: number;
  axonLengthUm: number;
  dendriteLengthUm: number;
  /** Soma depth below the pia, µm. See `piaDepth` for how the pia is placed. */
  depthBelowPiaUm: number;
  /** Synapses this neuron makes / receives, or null when the dataset has no synapse table. */
  outgoing: number | null;
  incoming: number | null;
}

export interface DatasetSummary {
  neuronCount: number;
  /** null while the synapse pass has not run. */
  synapseCount: number | null;
  excitatory: number;
  inhibitory: number;
  /** Cell types present, most common first. */
  perType: { code: string; name: string; count: number }[];
  /** Total skeleton cable of every neuron, kilometres. */
  totalCableKm: number;
  lod: Dataset["lod"];
}

export function layerName(layer: number): string {
  return layer >= 1 && layer <= 6 ? `Layer ${layer}` : "Layer unknown";
}

interface Cached {
  cablePerNeuron: Float64Array;
  axonPerNeuron: Float64Array;
  dendritePerNeuron: Float64Array;
  incoming: Uint32Array | null;
  pia: { y: number; source: "layer-fit" | "soma-min" };
}

const cache = new WeakMap<Dataset, Cached>();

/**
 * The depth (y, µm) of the pia. The scene already fits the layer boundaries from the somas
 * (`layerBoundaries`, pia first), so that fit is used. If no soma has a layer, fall back to the
 * shallowest soma minus 100 µm.
 */
export function piaDepth(data: Dataset): { y: number; source: "layer-fit" | "soma-min" } {
  return cached(data).pia;
}

function cached(data: Dataset): Cached {
  const hit = cache.get(data);
  if (hit) return hit;

  const { nodes, neurons, manifest } = data;
  const { min, max } = manifest.boundsUm;
  const cable = new Float64Array(neurons.count);
  const axon = new Float64Array(neurons.count);
  const dendrite = new Float64Array(neurons.count);
  for (let i = 0; i < nodes.count; i++) {
    const p = nodes.parent[i];
    if (p === NO_PARENT) continue;
    const dx = dequantise(nodes.pos[i * 3], min[0], max[0]) - dequantise(nodes.pos[p * 3], min[0], max[0]);
    const dy = dequantise(nodes.pos[i * 3 + 1], min[1], max[1]) - dequantise(nodes.pos[p * 3 + 1], min[1], max[1]);
    const dz = dequantise(nodes.pos[i * 3 + 2], min[2], max[2]) - dequantise(nodes.pos[p * 3 + 2], min[2], max[2]);
    const len = Math.hypot(dx, dy, dz);
    const n = nodes.neuronOfNode[i];
    cable[n] += len;
    if (nodes.compartment[i] === Compartment.Axon) axon[n] += len;
    else if (nodes.compartment[i] === Compartment.Dendrite) dendrite[n] += len;
  }

  let incoming: Uint32Array | null = null;
  if (data.synapses) {
    incoming = new Uint32Array(neurons.count);
    for (let s = 0; s < data.synapses.count; s++) incoming[data.synapses.post[s]]++;
  }

  const bounds = layerBoundaries(data);
  let pia: Cached["pia"];
  if (bounds.length > 0) {
    pia = { y: bounds[0], source: "layer-fit" };
  } else {
    let shallow = Infinity;
    for (let i = 0; i < neurons.count; i++) shallow = Math.min(shallow, neurons.somaUm[i * 3 + 1]);
    pia = { y: (Number.isFinite(shallow) ? shallow : 0) - 100, source: "soma-min" };
  }

  const entry = { cablePerNeuron: cable, axonPerNeuron: axon, dendritePerNeuron: dendrite, incoming, pia };
  cache.set(data, entry);
  return entry;
}

export function describeNeuron(data: Dataset, index: number): NeuronInfo {
  const { neurons, synapses } = data;
  if (!(index >= 0 && index < neurons.count)) throw new RangeError(`No neuron ${index} in a dataset of ${neurons.count}`);
  const c = cached(data);
  const typeCode = data.manifest.cellTypes[neurons.cellType[index]] ?? "";
  const layer = neurons.layer[index];
  return {
    index,
    typeCode,
    typeName: cellTypeName(typeCode),
    className: neurons.inhibitory[index] ? "inhibitory" : "excitatory",
    layer,
    layerName: layerName(layer),
    rootId: neurons.rootId[index].toString(),
    nodeCount: data.neuronNodeCount[index],
    cableLengthUm: c.cablePerNeuron[index],
    axonLengthUm: c.axonPerNeuron[index],
    dendriteLengthUm: c.dendritePerNeuron[index],
    depthBelowPiaUm: Math.max(0, neurons.somaUm[index * 3 + 1] - c.pia.y),
    outgoing: synapses ? synapses.preOffsets[index + 1] - synapses.preOffsets[index] : null,
    incoming: c.incoming ? c.incoming[index] : null,
  };
}

export function datasetSummary(data: Dataset): DatasetSummary {
  const { neurons, manifest } = data;
  const counts = new Map<string, number>();
  let inhibitory = 0;
  for (let i = 0; i < neurons.count; i++) {
    const code = manifest.cellTypes[neurons.cellType[i]] ?? "";
    counts.set(code, (counts.get(code) ?? 0) + 1);
    if (neurons.inhibitory[i]) inhibitory++;
  }
  const c = cached(data);
  let totalUm = 0;
  for (const v of c.cablePerNeuron) totalUm += v;
  return {
    neuronCount: neurons.count,
    synapseCount: data.synapses ? data.synapses.count : null,
    excitatory: neurons.count - inhibitory,
    inhibitory,
    perType: [...counts]
      .map(([code, count]) => ({ code, name: cellTypeName(code), count }))
      .sort((a, b) => b.count - a.count || a.code.localeCompare(b.code)),
    totalCableKm: totalUm / 1e9,
    lod: data.lod,
  };
}

const intFormat = new Intl.NumberFormat("en-US");

export function formatCount(n: number): string {
  return intFormat.format(Math.round(n));
}

/** 340 µm, 1.24 mm, 3.10 m: the unit that keeps the number readable. */
export function formatLength(um: number): string {
  if (um < 1000) return `${intFormat.format(Math.round(um))} µm`;
  if (um < 1e6) return `${(um / 1000).toFixed(um < 1e4 ? 2 : 1)} mm`;
  if (um < 1e9) return `${(um / 1e6).toFixed(2)} m`;
  return `${(um / 1e9).toFixed(2)} km`;
}
