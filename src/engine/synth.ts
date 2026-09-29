/**
 * A synthetic stand-in for the packed MICrONS dataset: random branching trees with roughly the
 * shape of cortical neurons, returned as the same `Dataset` that `loadDataset` produces, so the
 * renderer cannot tell the two apart. Deterministic for a given seed.
 *
 * Coordinates follow the real volume's convention: y is cortical depth, pia at low y.
 */

import { FORMAT_VERSION, NO_PARENT, PATH_DIST_UNIT_UM, quantise, type Bounds, type Manifest } from "./format";
import type { Dataset, NodeArrays } from "./data";

const SOMA = 0;
const AXON = 1;
const DENDRITE = 2;

const BOX: Bounds = { min: [0, 0, 0], max: [1000, 800, 500] };

/** Cell types in the order the manifest lists them; the last four are inhibitory. */
const CELL_TYPES = ["23P", "4P", "5P-IT", "5P-ET", "6P-IT", "BC", "MC", "BPC", "NGC"];
const FIRST_INHIBITORY = 5;

/** Lower y edge of layers 1..6 (index 0 is the pia), roughly the MICrONS laminar depths. */
const LAYER_BOTTOM = [0, 100, 200, 330, 450, 640, 800];

type Vec3 = [number, number, number];

/** mulberry32: small, fast, good enough for geometry and spike schedules. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function normalise(v: Vec3): Vec3 {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

function randomUnit(r: () => number): Vec3 {
  const z = r() * 2 - 1;
  const a = r() * Math.PI * 2;
  const s = Math.sqrt(1 - z * z);
  return [s * Math.cos(a), z, s * Math.sin(a)];
}

/** Rotate `dir` away from itself by `angle` radians in a random plane. */
function deflect(dir: Vec3, angle: number, r: () => number): Vec3 {
  const u = randomUnit(r);
  const d = u[0] * dir[0] + u[1] * dir[1] + u[2] * dir[2];
  const perp = normalise([u[0] - d * dir[0], u[1] - d * dir[1], u[2] - d * dir[2]]);
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return normalise([dir[0] * c + perp[0] * s, dir[1] * c + perp[1] * s, dir[2] * c + perp[2] * s]);
}

interface Branch {
  parent: number;
  pos: Vec3;
  dir: Vec3;
  /** Path length already covered from the soma centre, µm. */
  path: number;
  /** Remaining length of this branch, µm. */
  length: number;
  compartment: number;
  /** Radius in µm at the start of the branch. */
  radius: number;
  depth: number;
  /** Pull added to the direction each step, e.g. toward the pia for an apical dendrite. */
  bias: Vec3;
}

interface GrowParams {
  step: number;
  wiggle: number;
  branchProb: number;
  maxDepth: number;
  /** Radius every branch tapers toward, µm. */
  tipRadius: number;
  /** Path length over which a branch reaches the tip radius, µm. */
  taperLength: number;
  /** Most nodes this call may add. */
  budget: number;
}

class NeuronBuilder {
  x: number[] = [];
  y: number[] = [];
  z: number[] = [];
  radius: number[] = [];
  parent: number[] = [];
  path: number[] = [];
  comp: number[] = [];

  add(p: Vec3, radiusUm: number, parent: number, path: number, comp: number): number {
    this.x.push(p[0]);
    this.y.push(p[1]);
    this.z.push(p[2]);
    this.radius.push(radiusUm);
    this.parent.push(parent);
    this.path.push(path);
    this.comp.push(comp);
    return this.x.length - 1;
  }

  get count(): number {
    return this.x.length;
  }
}

/** Keep a growing tip inside the box by reflecting its direction off the walls. */
function reflect(pos: Vec3, dir: Vec3): Vec3 {
  const out: Vec3 = [dir[0], dir[1], dir[2]];
  const margin = 4;
  for (let k = 0; k < 3; k++) {
    if (pos[k] < BOX.min[k] + margin && out[k] < 0) out[k] = -out[k];
    if (pos[k] > BOX.max[k] - margin && out[k] > 0) out[k] = -out[k];
  }
  return out;
}

/** Grows branches breadth-first from `roots`, spawning children as it goes, until the budget runs out. */
function grow(nb: NeuronBuilder, roots: Branch[], p: GrowParams, r: () => number): void {
  const queue = [...roots];
  let used = 0;
  let head = 0;
  while (head < queue.length && used < p.budget) {
    const b = queue[head++];
    let { pos, dir } = b;
    let parent = b.parent;
    let path = b.path;
    let remaining = b.length;
    const startPath = b.path;
    while (remaining > 0 && used < p.budget) {
      const jitter = randomUnit(r);
      dir = normalise([
        dir[0] + jitter[0] * p.wiggle + b.bias[0],
        dir[1] + jitter[1] * p.wiggle + b.bias[1],
        dir[2] + jitter[2] * p.wiggle + b.bias[2],
      ]);
      dir = reflect(pos, dir);
      pos = [pos[0] + dir[0] * p.step, pos[1] + dir[1] * p.step, pos[2] + dir[2] * p.step];
      for (let k = 0; k < 3; k++) pos[k] = Math.min(BOX.max[k], Math.max(BOX.min[k], pos[k]));
      path += p.step;
      remaining -= p.step;
      const t = Math.min(1, (path - startPath) / p.taperLength);
      const radius = b.radius + (p.tipRadius - b.radius) * t;
      parent = nb.add(pos, radius, parent, path, b.compartment);
      used++;
      if (b.depth < p.maxDepth && remaining > p.step * 6 && r() < p.branchProb) {
        queue.push({
          parent,
          pos,
          dir: deflect(dir, 0.5 + r() * 0.7, r),
          path,
          length: remaining * (0.45 + r() * 0.5),
          compartment: b.compartment,
          radius: Math.max(p.tipRadius, radius * 0.75),
          depth: b.depth + 1,
          bias: b.bias,
        });
        dir = deflect(dir, 0.2 + r() * 0.3, r);
      }
    }
  }
}

function layerOfY(y: number): number {
  for (let l = 1; l <= 6; l++) if (y < LAYER_BOTTOM[l]) return l;
  return 6;
}

function pickLayer(r: () => number): number {
  const weights = [0, 0, 0.16, 0.2, 0.2, 0.24, 0.2];
  let u = r();
  for (let l = 2; l <= 6; l++) {
    u -= weights[l];
    if (u <= 0) return l;
  }
  return 6;
}

function excitatoryType(layer: number, r: () => number): number {
  if (layer <= 3) return 0;
  if (layer === 4) return 1;
  if (layer === 5) return r() < 0.6 ? 2 : 3;
  return 4;
}

function buildNeuron(nb: NeuronBuilder, soma: Vec3, inhibitory: boolean, r: () => number): void {
  const somaRadius = 5 + r() * 2.5;
  const somaIndex = nb.add(soma, somaRadius, -1, 0, SOMA);
  const nodeTarget = 2200 + Math.floor(r() * 600);

  // Axon: one long thin wandering process with a few collaterals. Pyramidal axons head down
  // toward the white matter first; interneuron axons set off in any direction and meander.
  const axonDir: Vec3 = inhibitory ? randomUnit(r) : normalise([(r() - 0.5) * 0.4, 1, (r() - 0.5) * 0.4]);
  const axonStart: Vec3 = [
    soma[0] + axonDir[0] * somaRadius,
    soma[1] + axonDir[1] * somaRadius,
    soma[2] + axonDir[2] * somaRadius,
  ];
  const axonRoot = nb.add(axonStart, 0.15, somaIndex, somaRadius, AXON);
  grow(
    nb,
    [
      {
        parent: axonRoot,
        pos: axonStart,
        dir: axonDir,
        path: somaRadius,
        length: 850 + r() * 450,
        compartment: AXON,
        radius: 0.15,
        depth: 0,
        bias: [0, 0, 0],
      },
    ],
    {
      step: 2.5,
      wiggle: inhibitory ? 0.22 : 0.1,
      branchProb: inhibitory ? 0.012 : 0.006,
      maxDepth: 2,
      tipRadius: 0.15,
      taperLength: 1,
      budget: Math.floor(nodeTarget * 0.32),
    },
    r,
  );

  // Dendrites share what is left of the node budget.
  const dendriteBudget = nodeTarget - nb.count;
  const roots: Branch[] = [];
  const addRoot = (dir: Vec3, length: number, radius: number, bias: Vec3) => {
    const start: Vec3 = [
      soma[0] + dir[0] * somaRadius,
      soma[1] + dir[1] * somaRadius,
      soma[2] + dir[2] * somaRadius,
    ];
    const idx = nb.add(start, radius, somaIndex, somaRadius, DENDRITE);
    roots.push({ parent: idx, pos: start, dir, path: somaRadius, length, compartment: DENDRITE, radius, depth: 0, bias });
  };
  if (!inhibitory) {
    // Apical dendrite climbs toward the pia and tufts out in layer 1.
    const reach = Math.max(60, soma[1] - 20);
    addRoot(normalise([(r() - 0.5) * 0.2, -1, (r() - 0.5) * 0.2]), reach + 120, 1.0, [0, -0.06, 0]);
    const basal = 5 + Math.floor(r() * 3);
    for (let i = 0; i < basal; i++) {
      const d = randomUnit(r);
      addRoot(normalise([d[0], Math.abs(d[1]) * 0.6, d[2]]), 120 + r() * 110, 0.7 + r() * 0.2, [0, 0, 0]);
    }
  } else {
    const count = 6 + Math.floor(r() * 4);
    for (let i = 0; i < count; i++) addRoot(randomUnit(r), 100 + r() * 90, 0.6 + r() * 0.25, [0, 0, 0]);
  }
  grow(
    nb,
    roots,
    {
      step: 1.6,
      wiggle: 0.2,
      branchProb: inhibitory ? 0.05 : 0.035,
      maxDepth: 5,
      tipRadius: 0.3,
      taperLength: 160,
      budget: dendriteBudget,
    },
    r,
  );
}

/** Builds `neuronCount` random neurons (about 2,500 nodes each) inside a 1000 x 800 x 500 µm box. */
export function synthDataset(seed: number, neuronCount: number): Dataset {
  const r = rng(seed);
  const builders: NeuronBuilder[] = [];
  const somaUm = new Float32Array(neuronCount * 3);
  const cellType = new Uint8Array(neuronCount);
  const inhibitory = new Uint8Array(neuronCount);
  const layer = new Uint8Array(neuronCount);
  const rootId = new BigUint64Array(neuronCount);

  for (let i = 0; i < neuronCount; i++) {
    const inh = r() < 0.18;
    const l = pickLayer(r);
    const top = LAYER_BOTTOM[l - 1];
    const soma: Vec3 = [60 + r() * 880, top + 10 + r() * (LAYER_BOTTOM[l] - top - 20), 40 + r() * 420];
    const nb = new NeuronBuilder();
    buildNeuron(nb, soma, inh, r);
    builders.push(nb);
    somaUm.set(soma, i * 3);
    inhibitory[i] = inh ? 1 : 0;
    cellType[i] = inh
      ? FIRST_INHIBITORY + Math.floor(r() * (CELL_TYPES.length - FIRST_INHIBITORY))
      : excitatoryType(l, r);
    layer[i] = layerOfY(soma[1]);
    rootId[i] = 864691135000000000n + BigInt(i);
  }

  const total = builders.reduce((s, b) => s + b.count, 0);
  const nodes: NodeArrays = {
    count: total,
    pos: new Uint16Array(total * 3),
    radiusNm: new Uint16Array(total),
    parent: new Uint32Array(total),
    pathDistQ: new Uint16Array(total),
    compartment: new Uint8Array(total),
    neuronOfNode: new Uint16Array(total),
  };
  const neuronNodeStart = new Uint32Array(neuronCount);
  const neuronNodeCount = new Uint32Array(neuronCount);

  let base = 0;
  builders.forEach((nb, n) => {
    neuronNodeStart[n] = base;
    neuronNodeCount[n] = nb.count;
    for (let i = 0; i < nb.count; i++) {
      const g = base + i;
      nodes.pos[g * 3] = quantise(nb.x[i], BOX.min[0], BOX.max[0]);
      nodes.pos[g * 3 + 1] = quantise(nb.y[i], BOX.min[1], BOX.max[1]);
      nodes.pos[g * 3 + 2] = quantise(nb.z[i], BOX.min[2], BOX.max[2]);
      nodes.radiusNm[g] = Math.min(65535, Math.round(nb.radius[i] * 1000));
      nodes.parent[g] = nb.parent[i] < 0 ? NO_PARENT : base + nb.parent[i];
      nodes.pathDistQ[g] = Math.min(65535, Math.round(nb.path[i] / PATH_DIST_UNIT_UM));
      nodes.compartment[g] = nb.comp[i];
      nodes.neuronOfNode[g] = n;
    }
    base += nb.count;
  });

  const lod = { nodeCount: total, chunks: [] };
  const manifest: Manifest = {
    version: FORMAT_VERSION,
    boundsUm: BOX,
    neuronCount,
    cellTypes: CELL_TYPES,
    neurons: "neurons.bin",
    lods: { hi: lod, lite: lod },
    synapses: null,
    credits: {
      dataset: "Synthetic stand-in (not real neurons)",
      licence: "n/a",
      url: "",
      citations: [],
    },
  };

  return {
    manifest,
    lod: "hi",
    neurons: { count: neuronCount, rootId, somaUm, cellType, inhibitory, layer },
    nodes,
    neuronNodeStart,
    neuronNodeCount,
    synapses: null,
  };
}
