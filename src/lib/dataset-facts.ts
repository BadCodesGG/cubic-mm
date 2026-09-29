/**
 * Numbers about the dataset for server-rendered pages, read from public/data at build time so
 * the prose never drifts from what the site actually loads.
 */

import { readFileSync } from "node:fs";
import path from "node:path";
import manifest from "../../public/data/manifest.json";
import { decodeNeurons, parseManifest } from "@/engine/format";

/** Same codes as the HUD's names (src/engine/info.ts, which cannot be imported without three). */
const CELL_TYPE_NAMES: Record<string, string> = {
  "23P": "Layer 2/3 pyramidal",
  "4P": "Layer 4 pyramidal",
  "5P-IT": "Layer 5 IT pyramidal",
  "5P-ET": "Layer 5 ET pyramidal",
  "5P-NP": "Layer 5 near-projecting pyramidal",
  "6P-IT": "Layer 6 IT pyramidal",
  "6P-CT": "Layer 6 corticothalamic pyramidal",
  "6P-U": "Layer 6 pyramidal, unclassified",
  BC: "Basket cell",
  MC: "Martinotti cell",
  BPC: "Bipolar cell",
  NGC: "Neurogliaform cell",
  "": "Unclassified",
};

export interface DatasetFacts {
  neuronCount: number;
  synapseCount: number;
  /** Extent of the volume in µm along x, y (depth) and z. */
  extentUm: [number, number, number];
  cellTypes: { code: string; name: string; count: number }[];
  excitatory: number;
  inhibitory: number;
  credits: { dataset: string; licence: string; url: string; citations: string[] };
}

export function datasetFacts(): DatasetFacts {
  const m = parseManifest(manifest);
  const buf = readFileSync(path.join(process.cwd(), "public", "data", m.neurons));
  const neurons = decodeNeurons(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer);
  const counts = new Map<string, number>();
  let inhibitory = 0;
  for (let i = 0; i < neurons.count; i++) {
    const code = m.cellTypes[neurons.cellType[i]] ?? "";
    counts.set(code, (counts.get(code) ?? 0) + 1);
    if (neurons.inhibitory[i]) inhibitory++;
  }
  const { min, max } = m.boundsUm;
  return {
    neuronCount: m.neuronCount,
    synapseCount: m.synapses?.count ?? 0,
    extentUm: [max[0] - min[0], max[1] - min[1], max[2] - min[2]],
    cellTypes: [...counts]
      .map(([code, count]) => ({ code, name: CELL_TYPE_NAMES[code] ?? code, count }))
      .sort((a, b) => b.count - a.count || a.code.localeCompare(b.code)),
    excitatory: neurons.count - inhibitory,
    inhibitory,
    credits: m.credits,
  };
}
