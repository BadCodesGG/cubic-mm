/**
 * Loads a packed dataset (see `format.ts`) into flat, GPU-ready arrays.
 *
 * Chunks are fetched in parallel and merged into one node index space per LOD, in the order the
 * manifest lists them, so a node's global index is `chunk.nodeStart + local`. Parent indices are
 * already global in the files, so no remapping happens here.
 */

import {
  NO_PARENT,
  decodeChunk,
  decodeNeurons,
  decodeSynapses,
  parseManifest,
  type LodName,
  type Manifest,
  type NeuronTable,
  type SynapseTable,
} from "./format";

export interface NodeArrays {
  count: number;
  /** Quantised xyz, 3 per node; dequantise against `manifest.boundsUm`. */
  pos: Uint16Array;
  radiusNm: Uint16Array;
  /** Global parent index or NO_PARENT. */
  parent: Uint32Array;
  pathDistQ: Uint16Array;
  compartment: Uint8Array;
  /** Owning neuron index, per node. */
  neuronOfNode: Uint16Array;
}

export interface Dataset {
  manifest: Manifest;
  lod: LodName;
  neurons: NeuronTable;
  nodes: NodeArrays;
  /** First global node index of each neuron (its soma). */
  neuronNodeStart: Uint32Array;
  neuronNodeCount: Uint32Array;
  synapses: SynapseTable | null;
}

export type ProgressFn = (loadedBytes: number, totalBytes: number, label: string) => void;

async function fetchBuffer(url: string, signal?: AbortSignal): Promise<ArrayBuffer> {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.arrayBuffer();
}

/**
 * @param baseUrl Directory holding manifest.json, e.g. "/data".
 * @param lod Which LOD's chunks to load. The renderer picks "lite" on WebGL2 or a weak device.
 */
export async function loadDataset(
  baseUrl: string,
  lod: LodName,
  onProgress: ProgressFn = () => {},
  signal?: AbortSignal,
): Promise<Dataset> {
  const base = baseUrl.replace(/\/$/, "");
  onProgress(0, 1, "manifest");
  const manifestRes = await fetch(`${base}/manifest.json`, { signal });
  if (!manifestRes.ok) throw new Error(`${base}/manifest.json: HTTP ${manifestRes.status}`);
  const manifest = parseManifest(await manifestRes.json());
  const lodInfo = manifest.lods[lod];

  const neurons = decodeNeurons(await fetchBuffer(`${base}/${manifest.neurons}`, signal));

  const total = lodInfo.chunks.length + (manifest.synapses ? 1 : 0);
  let done = 0;
  const tick = (label: string) => onProgress(++done, total, label);

  const nodes: NodeArrays = {
    count: lodInfo.nodeCount,
    pos: new Uint16Array(lodInfo.nodeCount * 3),
    radiusNm: new Uint16Array(lodInfo.nodeCount),
    parent: new Uint32Array(lodInfo.nodeCount).fill(NO_PARENT),
    pathDistQ: new Uint16Array(lodInfo.nodeCount),
    compartment: new Uint8Array(lodInfo.nodeCount),
    neuronOfNode: new Uint16Array(lodInfo.nodeCount),
  };
  const neuronNodeStart = new Uint32Array(neurons.count);
  const neuronNodeCount = new Uint32Array(neurons.count);

  await Promise.all(
    lodInfo.chunks.map(async (info) => {
      const chunk = decodeChunk(await fetchBuffer(`${base}/${info.file}`, signal));
      if (chunk.nodeStart !== info.nodeStart || chunk.nodeCount !== info.nodeCount) {
        throw new Error(`${info.file}: header disagrees with the manifest`);
      }
      const s = chunk.nodeStart;
      nodes.pos.set(chunk.pos, s * 3);
      nodes.radiusNm.set(chunk.radiusNm, s);
      nodes.parent.set(chunk.parent, s);
      nodes.pathDistQ.set(chunk.pathDistQ, s);
      nodes.compartment.set(chunk.compartment, s);
      for (const n of chunk.neurons) {
        neuronNodeStart[n.neuronIndex] = n.nodeStart;
        neuronNodeCount[n.neuronIndex] = n.nodeCount;
        nodes.neuronOfNode.fill(n.neuronIndex, n.nodeStart, n.nodeStart + n.nodeCount);
      }
      tick(info.file);
    }),
  );

  let synapses: SynapseTable | null = null;
  if (manifest.synapses) {
    synapses = decodeSynapses(await fetchBuffer(`${base}/${manifest.synapses.file}`, signal));
    tick(manifest.synapses.file);
  }

  return { manifest, lod, neurons, nodes, neuronNodeStart, neuronNodeCount, synapses };
}
