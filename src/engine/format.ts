/**
 * The on-disk data format for One Cubic Millimetre. This file is the single source of truth:
 * `pipeline/pack.py` writes exactly these layouts and the tests here round-trip them through the
 * encoders below, so a change to either side that is not mirrored in the other fails a test.
 *
 * Every file is little-endian. Every array section starts at a multiple of 4 bytes so a typed
 * array can be viewed straight over the file's ArrayBuffer without copying. Sections are laid
 * out widest-first (u64, then u32/f32, then u16, then u8) so only the u8 tails ever need padding.
 *
 * Coordinates: the MICrONS volume in micrometres (the SWC frame). Positions are quantised to
 * uint16 against `manifest.boundsUm`; path distances are quarter-micrometres in uint16, so the
 * longest representable path is 16,383 µm, which is longer than any axon in the cube.
 */

export const FORMAT_VERSION = 1;

/** Sentinel for "this node is a root" in the parent array. */
export const NO_PARENT = 0xffffffff;

export const enum Compartment {
  Soma = 0,
  Axon = 1,
  Dendrite = 2,
}

export type LodName = "hi" | "lite";

export interface Bounds {
  min: [number, number, number];
  max: [number, number, number];
}

export interface ChunkInfo {
  /** Path relative to the manifest, e.g. "chunks/hi/0.bin". */
  file: string;
  /** Index of the first node of this chunk in the LOD's global node index space. */
  nodeStart: number;
  nodeCount: number;
  neuronCount: number;
  /** Axis-aligned bounds of the nodes in this chunk, in µm. */
  bboxUm: Bounds;
}

export interface LodInfo {
  nodeCount: number;
  chunks: ChunkInfo[];
}

export interface Manifest {
  version: number;
  /** Global quantisation bounds in µm. Every LOD and the synapse file share them. */
  boundsUm: Bounds;
  neuronCount: number;
  /** Index into this array is the `cellType` byte of a neuron. */
  cellTypes: string[];
  neurons: string;
  lods: Record<LodName, LodInfo>;
  /** Absent in round 0, before the synapse pass has run. */
  synapses: { file: string; count: number } | null;
  credits: {
    dataset: string;
    licence: string;
    url: string;
    citations: string[];
  };
}

/** Static per-neuron facts. Node ranges live in each LOD's chunk headers instead, since they differ per LOD. */
export interface NeuronTable {
  count: number;
  rootId: BigUint64Array;
  /** Soma position in µm, xyz interleaved. */
  somaUm: Float32Array;
  cellType: Uint8Array;
  /** 1 if inhibitory, 0 if excitatory. Drives the sign of every synapse the neuron makes. */
  inhibitory: Uint8Array;
  /** Cortical layer 1 to 6 by soma depth, 0 if unknown. */
  layer: Uint8Array;
}

export interface ChunkNeuronRange {
  neuronIndex: number;
  /** Global node index (within the LOD) of the neuron's first node; the soma is always first. */
  nodeStart: number;
  nodeCount: number;
}

export interface Chunk {
  nodeStart: number;
  nodeCount: number;
  neurons: ChunkNeuronRange[];
  /** Quantised xyz, 3 per node. Dequantise with `dequantise`. */
  pos: Uint16Array;
  /** Radius in nanometres. */
  radiusNm: Uint16Array;
  /** Global node index of the parent within the LOD, or NO_PARENT for the soma. */
  parent: Uint32Array;
  /** Path length from the soma along the tree, in quarter-µm. */
  pathDistQ: Uint16Array;
  compartment: Uint8Array;
}

export interface SynapseTable {
  count: number;
  neuronCount: number;
  /** preOffsets[n]..preOffsets[n+1] is the run of synapses whose presynaptic neuron is n. */
  preOffsets: Uint32Array;
  pre: Uint16Array;
  post: Uint16Array;
  /** Quantised synapse position, 3 per synapse. */
  pos: Uint16Array;
  /** Path distance from the presynaptic soma to the synapse along the axon, quarter-µm. */
  preDistQ: Uint16Array;
  /** Cleft size bucketed to 0..255 (log scale in the packer); a weight hint, not a physical unit. */
  size: Uint8Array;
}

const MAGIC = { neurons: "CMN1", chunk: "CMM1", synapses: "CMS1" } as const;

export const PATH_DIST_UNIT_UM = 0.25;

export function quantise(valueUm: number, min: number, max: number): number {
  const t = (valueUm - min) / (max - min);
  return Math.max(0, Math.min(65535, Math.round(t * 65535)));
}

export function dequantise(q: number, min: number, max: number): number {
  return min + (q / 65535) * (max - min);
}

function align4(n: number): number {
  return (n + 3) & ~3;
}

function align8(n: number): number {
  return (n + 7) & ~7;
}

function readMagic(view: DataView, expected: string): void {
  const got = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  if (got !== expected) throw new Error(`Bad magic: expected ${expected}, got ${got}`);
}

function writeMagic(view: DataView, magic: string): void {
  for (let i = 0; i < 4; i++) view.setUint8(i, magic.charCodeAt(i));
}

/** Little cursor over an ArrayBuffer that hands out aligned typed-array views. */
class Reader {
  offset: number;
  constructor(readonly buffer: ArrayBuffer, start: number) {
    this.offset = start;
  }
  u64(count: number): BigUint64Array {
    this.offset = align8(this.offset);
    const a = new BigUint64Array(this.buffer, this.offset, count);
    this.offset += count * 8;
    return a;
  }
  u32(count: number): Uint32Array {
    this.offset = align4(this.offset);
    const a = new Uint32Array(this.buffer, this.offset, count);
    this.offset += count * 4;
    return a;
  }
  f32(count: number): Float32Array {
    this.offset = align4(this.offset);
    const a = new Float32Array(this.buffer, this.offset, count);
    this.offset += count * 4;
    return a;
  }
  u16(count: number): Uint16Array {
    this.offset = align4(this.offset);
    const a = new Uint16Array(this.buffer, this.offset, count);
    this.offset += count * 2;
    return a;
  }
  u8(count: number): Uint8Array {
    this.offset = align4(this.offset);
    const a = new Uint8Array(this.buffer, this.offset, count);
    this.offset += count;
    return a;
  }
}

class Writer {
  private parts: { at: number; bytes: Uint8Array }[] = [];
  offset: number;
  constructor(start: number) {
    this.offset = start;
  }
  private put(bytes: Uint8Array, alignment: number): void {
    this.offset = alignment === 8 ? align8(this.offset) : align4(this.offset);
    this.parts.push({ at: this.offset, bytes });
    this.offset += bytes.byteLength;
  }
  u64(a: BigUint64Array): void {
    this.put(new Uint8Array(a.buffer, a.byteOffset, a.byteLength), 8);
  }
  u32(a: Uint32Array): void {
    this.put(new Uint8Array(a.buffer, a.byteOffset, a.byteLength), 4);
  }
  f32(a: Float32Array): void {
    this.put(new Uint8Array(a.buffer, a.byteOffset, a.byteLength), 4);
  }
  u16(a: Uint16Array): void {
    this.put(new Uint8Array(a.buffer, a.byteOffset, a.byteLength), 4);
  }
  u8(a: Uint8Array): void {
    this.put(a, 4);
  }
  finish(header: (view: DataView) => void): ArrayBuffer {
    const out = new ArrayBuffer(align4(this.offset));
    const bytes = new Uint8Array(out);
    for (const p of this.parts) bytes.set(p.bytes, p.at);
    header(new DataView(out));
    return out;
  }
}

/* ------------------------------------------------------------------ neurons.bin
 * magic "CMN1" u8[4] | count u32 | rootId u64[count] | somaUm f32[3*count]
 * | cellType u8[count] | inhibitory u8[count] | layer u8[count]
 */

export function decodeNeurons(buffer: ArrayBuffer): NeuronTable {
  const view = new DataView(buffer);
  readMagic(view, MAGIC.neurons);
  const count = view.getUint32(4, true);
  const r = new Reader(buffer, 8);
  return {
    count,
    rootId: r.u64(count),
    somaUm: r.f32(count * 3),
    cellType: r.u8(count),
    inhibitory: r.u8(count),
    layer: r.u8(count),
  };
}

export function encodeNeurons(t: NeuronTable): ArrayBuffer {
  const w = new Writer(8);
  w.u64(t.rootId);
  w.f32(t.somaUm);
  w.u8(t.cellType);
  w.u8(t.inhibitory);
  w.u8(t.layer);
  return w.finish((view) => {
    writeMagic(view, MAGIC.neurons);
    view.setUint32(4, t.count, true);
  });
}

/* ------------------------------------------------------------------ chunks/<lod>/<i>.bin
 * magic "CMM1" u8[4] | neuronCount u32 | nodeCount u32 | nodeStart u32
 * | neuronRanges u32[3*neuronCount] (neuronIndex, nodeStart, nodeCount)
 * | parent u32[nodeCount] | pos u16[3*nodeCount] | radiusNm u16[nodeCount]
 * | pathDistQ u16[nodeCount] | compartment u8[nodeCount]
 */

export function decodeChunk(buffer: ArrayBuffer): Chunk {
  const view = new DataView(buffer);
  readMagic(view, MAGIC.chunk);
  const neuronCount = view.getUint32(4, true);
  const nodeCount = view.getUint32(8, true);
  const nodeStart = view.getUint32(12, true);
  const r = new Reader(buffer, 16);
  const ranges = r.u32(neuronCount * 3);
  const neurons: ChunkNeuronRange[] = [];
  for (let i = 0; i < neuronCount; i++) {
    neurons.push({ neuronIndex: ranges[i * 3], nodeStart: ranges[i * 3 + 1], nodeCount: ranges[i * 3 + 2] });
  }
  return {
    nodeStart,
    nodeCount,
    neurons,
    parent: r.u32(nodeCount),
    pos: r.u16(nodeCount * 3),
    radiusNm: r.u16(nodeCount),
    pathDistQ: r.u16(nodeCount),
    compartment: r.u8(nodeCount),
  };
}

export function encodeChunk(c: Chunk): ArrayBuffer {
  const ranges = new Uint32Array(c.neurons.length * 3);
  c.neurons.forEach((n, i) => {
    ranges[i * 3] = n.neuronIndex;
    ranges[i * 3 + 1] = n.nodeStart;
    ranges[i * 3 + 2] = n.nodeCount;
  });
  const w = new Writer(16);
  w.u32(ranges);
  w.u32(c.parent);
  w.u16(c.pos);
  w.u16(c.radiusNm);
  w.u16(c.pathDistQ);
  w.u8(c.compartment);
  return w.finish((view) => {
    writeMagic(view, MAGIC.chunk);
    view.setUint32(4, c.neurons.length, true);
    view.setUint32(8, c.nodeCount, true);
    view.setUint32(12, c.nodeStart, true);
  });
}

/* ------------------------------------------------------------------ synapses.bin
 * magic "CMS1" u8[4] | count u32 | neuronCount u32
 * | preOffsets u32[neuronCount+1] | pre u16[count] | post u16[count]
 * | pos u16[3*count] | preDistQ u16[count] | size u8[count]
 * Sorted by pre, so preOffsets indexes runs.
 */

export function decodeSynapses(buffer: ArrayBuffer): SynapseTable {
  const view = new DataView(buffer);
  readMagic(view, MAGIC.synapses);
  const count = view.getUint32(4, true);
  const neuronCount = view.getUint32(8, true);
  const r = new Reader(buffer, 12);
  return {
    count,
    neuronCount,
    preOffsets: r.u32(neuronCount + 1),
    pre: r.u16(count),
    post: r.u16(count),
    pos: r.u16(count * 3),
    preDistQ: r.u16(count),
    size: r.u8(count),
  };
}

export function encodeSynapses(t: SynapseTable): ArrayBuffer {
  const w = new Writer(12);
  w.u32(t.preOffsets);
  w.u16(t.pre);
  w.u16(t.post);
  w.u16(t.pos);
  w.u16(t.preDistQ);
  w.u8(t.size);
  return w.finish((view) => {
    writeMagic(view, MAGIC.synapses);
    view.setUint32(4, t.count, true);
    view.setUint32(8, t.neuronCount, true);
  });
}

/* ------------------------------------------------------------------ manifest.json */

export function parseManifest(json: unknown): Manifest {
  const m = json as Manifest;
  if (!m || typeof m !== "object") throw new Error("Manifest is not an object");
  if (m.version !== FORMAT_VERSION) throw new Error(`Manifest version ${m.version}, expected ${FORMAT_VERSION}`);
  if (!m.lods?.hi || !m.lods?.lite) throw new Error("Manifest is missing an LOD");
  if (!Array.isArray(m.cellTypes)) throw new Error("Manifest is missing cellTypes");
  return m;
}
