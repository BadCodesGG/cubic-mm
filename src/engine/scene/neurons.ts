/**
 * The neurons: every skeleton edge (child to parent) is one instance of a camera-facing ribbon,
 * and every soma one instance of a sphere. Both are additive light, so the volume reads as a
 * haze of faint structure that brightens only where it is dense, near, or carrying a spike.
 *
 * Per-neuron data is indexed by neuron number and read in the vertex stage:
 *   info:   a float texture, one texel per neuron, rgb = base tint (linear), a = brightness
 *   spikes: the last 8 spike times in seconds (ring buffer). With the GPU simulation this is the
 *           simulation's own storage buffer, read directly; otherwise a float texture of two
 *           texels per neuron that `update()` refreshes from the CPU worker. `spikeSlots` is the
 *           one accessor both paths go through, so the shaders are written once.
 *
 * On top of the ribbons: a point of light riding each pulse front along the neuron's longest
 * axon branch, and a brief glow on the dendrite at each synapse as a pulse lands there.
 */

import {
  AdditiveBlending,
  CustomBlending,
  OneFactor,
  OneMinusSrcAlphaFactor,
  Color,
  DataTexture,
  FloatType,
  InstancedBufferAttribute,
  InstancedBufferGeometry,
  InstancedMesh,
  Float32BufferAttribute,
  Matrix4,
  Mesh,
  MeshBasicNodeMaterial,
  NearestFilter,
  RGBAFormat,
  SphereGeometry,
  Vector3,
} from "three/webgpu";
import {
  abs,
  attribute,
  clamp,
  cameraProjectionMatrix,
  cross,
  exp,
  float,
  floor,
  instanceIndex,
  ivec2,
  length,
  max,
  min,
  mix,
  mod,
  modelViewMatrix,
  normalView,
  positionLocal,
  positionView,
  pow,
  select,
  smoothstep,
  sqrt,
  step,
  storage,
  textureLoad,
  uint,
  varying,
  vec3,
  vec4,
} from "three/tsl";
import type { Node } from "three/webgpu";
import { NO_PARENT, PATH_DIST_UNIT_UM, dequantise, type SynapseTable } from "../format";
import type { Dataset } from "../data";
import { rng } from "../synth";
import { EMPTY_SPIKE, SPIKE_SLOTS } from "../sim/model";
import type { SpikeTimesSource } from "../sim";
import type { SceneUniforms } from "./uniforms";

export { SPIKE_SLOTS };
/** Spike slots start here: long enough ago that nothing glows. */
export const NEVER = EMPTY_SPIKE;
/** Samples of each neuron's longest axon branch, evenly spaced in path length, for the pulse sprites. */
const PATH_SAMPLES = 128;
const TEX_WIDTH = 1024;
/** Longest piece a skeleton edge is drawn with, µm. */
const PIECE_UM = 3;
/** Typical soma radius, µm, for sizing its glow on screen. */
const SOMA_GLOW_UM = 3.5;

type FloatNode = Node<"float">;

export interface NeuronTextures {
  info: DataTexture;
  spikes: DataTexture;
  /** View over `spikes`' data: neuron n's slots are [n * 8, n * 8 + 8). */
  spikeTimes: Float32Array;
}

function makeTexture(texels: number): DataTexture {
  const height = Math.max(1, Math.ceil(texels / TEX_WIDTH));
  const data = new Float32Array(TEX_WIDTH * height * 4);
  const tex = new DataTexture(data, TEX_WIDTH, height, RGBAFormat, FloatType);
  tex.magFilter = NearestFilter;
  tex.minFilter = NearestFilter;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}

/** Texel coordinate of the `k`th texel of a record of `stride` texels for record `index`. */
function texel(index: FloatNode, stride: number, k: number | FloatNode) {
  const t = index.mul(stride).add(k);
  return ivec2(mod(t, TEX_WIDTH).toInt(), floor(t.div(TEX_WIDTH)).toInt());
}

function createNeuronTextures(data: Dataset): NeuronTextures {
  const n = data.neurons.count;
  const info = makeTexture(n);
  const spikes = makeTexture(n * 2);
  const infoData = info.image.data as Float32Array;
  const spikeTimes = spikes.image.data as Float32Array;
  spikeTimes.fill(NEVER);

  const r = rng(7);
  const c = new Color();
  for (let i = 0; i < n; i++) {
    const inhibitory = data.neurons.inhibitory[i] === 1;
    // Excitatory cells sit in a narrow cool band from teal to cyan-blue; inhibitory cells in a
    // violet-to-orchid band. Low saturation keeps it biological rather than neon.
    if (inhibitory) c.setHSL(0.74 + r() * 0.12, 0.5 + r() * 0.15, 0.55);
    else c.setHSL(0.46 + r() * 0.12, 0.55 + r() * 0.2, 0.46);
    // Most cells are dim and a few are bright, so the tangle never reads as one flat colour.
    const v = r();
    infoData.set([c.r, c.g, c.b, 0.1 + v * v * v * 1.25], i * 4);
  }
  return { info, spikes, spikeTimes };
}

type SlotPair = [Node<"vec4">, Node<"vec4">];
/** Reads one neuron's 8 spike slots as two vec4s, from whichever store the simulation feeds. */
type SpikeSlots = (neuron: FloatNode) => SlotPair;

function spikeSlots(tex: NeuronTextures, source: SpikeTimesSource | undefined): SpikeSlots {
  if (source?.kind === "storage") {
    const buf = storage(source.attribute, "float", source.attribute.count).toReadOnly();
    return (neuron) => {
      const base = neuron.toUint().mul(SPIKE_SLOTS);
      const at = (k: number) => buf.element(base.add(k));
      return [vec4(at(0), at(1), at(2), at(3)), vec4(at(4), at(5), at(6), at(7))];
    };
  }
  return (neuron) => [textureLoad(tex.spikes, texel(neuron, 2, 0)), textureLoad(tex.spikes, texel(neuron, 2, 1))];
}

/**
 * Spike light falls off with view depth on top of the haze, so near activity is what reads. The
 * stimulated neuron falls off more gently, so its cascade stays bright across the frame.
 */
function spikeNear(u: SceneUniforms, depth: FloatNode, neuron: FloatNode) {
  const x = depth.div(u.spikeNearUm);
  const x2 = x.mul(x);
  const y = depth.div(u.cascadeNearUm);
  return select(neuron.equal(u.stimNeuron), float(1).div(y.mul(y).add(1)), float(1).div(x2.mul(x2).add(1)));
}

/** Sum of the spike pulse and its afterglow over one neuron's 8 slots, at path distance `path` µm. */
function spikeGlow(
  u: SceneUniforms,
  slotsA: Node<"vec4">,
  slotsB: Node<"vec4">,
  path: FloatNode,
  afterglow: FloatNode = u.afterglow,
) {
  const delay = path.mul(u.slowMo).mul(1e-6).div(u.velocity);
  let pulse: FloatNode = float(0);
  let after: FloatNode = float(0);
  for (const s of [slotsA.x, slotsA.y, slotsA.z, slotsA.w, slotsB.x, slotsB.y, slotsB.z, slotsB.w]) {
    const dt = u.time.sub(s.add(delay));
    const x = dt.div(u.pulseWidth);
    pulse = pulse.add(exp(x.mul(x).negate()));
    after = after.add(step(0, dt).mul(exp(dt.negate().div(afterglow))));
  }
  return { pulse, after };
}

/** First child of each node (the one its tangent continues into), or -1 for a leaf. */
function firstChildren(count: number, parent: Uint32Array): Int32Array {
  const first = new Int32Array(count).fill(-1);
  for (let i = 0; i < count; i++) {
    const p = parent[i];
    if (p !== NO_PARENT && first[p] < 0) first[p] = i;
  }
  return first;
}

/**
 * Unit tangent at every node, pointing toward the soma: the average of the direction in from
 * its first child and the direction on to its parent.
 */
function nodeTangents(count: number, parent: Uint32Array, compartment: Uint8Array, pos: Float32Array): Float32Array {
  const first = firstChildren(count, parent);
  const out = new Float32Array(count * 3);
  const dir = (from: number, to: number, w: number, acc: number[]) => {
    const dx = pos[to * 3] - pos[from * 3];
    const dy = pos[to * 3 + 1] - pos[from * 3 + 1];
    const dz = pos[to * 3 + 2] - pos[from * 3 + 2];
    const l = Math.hypot(dx, dy, dz) || 1;
    acc[0] += (dx / l) * w;
    acc[1] += (dy / l) * w;
    acc[2] += (dz / l) * w;
  };
  const acc = [0, 0, 0];
  for (let i = 0; i < count; i++) {
    acc[0] = acc[1] = acc[2] = 0;
    const p = parent[i];
    if (p !== NO_PARENT) dir(i, p, 1, acc);
    const c = first[i];
    if (c >= 0 && compartment[i] !== 0) dir(c, i, 1, acc);
    const l = Math.hypot(acc[0], acc[1], acc[2]);
    if (l > 1e-6) out.set([acc[0] / l, acc[1] / l, acc[2] / l], i * 3);
    else out.set([1, 0, 0], i * 3);
  }
  return out;
}

export interface Neurons {
  ribbons: Mesh;
  somas: InstancedMesh;
  /** A point of light riding every pulse front along each neuron's longest axon branch. */
  pulses: Mesh;
  /** A brief glow at each synapse as a pulse lands on it; null without a synapse table. */
  synapseGlow: Mesh | null;
  /** Uploads new spike times when they come from the CPU worker; a no-op on the GPU path. */
  update(): void;
  textures: NeuronTextures;
  instanceCount: number;
  /** World-space soma centres, µm. */
  somaPos: Float32Array;
  dispose(): void;
}

/**
 * @param maxPieces Most pieces one skeleton edge is split into for smooth curves; lower it on
 *   weaker backends, since it multiplies the instance count.
 */
export interface NeuronOptions {
  /** Where spike times come from. Without one they stay empty and nothing fires. */
  spikes?: SpikeTimesSource;
  /** Synapses to draw arrival glows at. */
  synapses?: SynapseTable | null;
}

export function createNeurons(data: Dataset, u: SceneUniforms, maxPieces = 3, opts: NeuronOptions = {}): Neurons {
  const { nodes, manifest } = data;
  const { min: bmin, max: bmax } = manifest.boundsUm;
  const textures = createNeuronTextures(data);
  const slots = spikeSlots(textures, opts.spikes);

  // Dequantise every node once.
  const pos = new Float32Array(nodes.count * 3);
  for (let i = 0; i < nodes.count; i++) {
    for (let k = 0; k < 3; k++) pos[i * 3 + k] = dequantise(nodes.pos[i * 3 + k], bmin[k], bmax[k]);
  }

  const tangents = nodeTangents(nodes.count, nodes.parent, nodes.compartment, pos);
  const mainChild = firstChildren(nodes.count, nodes.parent);

  // Skeleton edges are several µm long, which reads as a zig-zag up close. Each edge is drawn
  // as up to `maxPieces` pieces along a cubic Hermite curve through the node tangents, so
  // branches bend smoothly and neighbouring edges still meet exactly.
  const pieces = (i: number, p: number) => {
    if (nodes.compartment[p] === 0) return 1;
    const dx = pos[p * 3] - pos[i * 3];
    const dy = pos[p * 3 + 1] - pos[i * 3 + 1];
    const dz = pos[p * 3 + 2] - pos[i * 3 + 2];
    return Math.min(maxPieces, Math.max(1, Math.ceil(Math.hypot(dx, dy, dz) / PIECE_UM)));
  };
  let edges = 0;
  for (let i = 0; i < nodes.count; i++) {
    const p = nodes.parent[i];
    if (p !== NO_PARENT) edges += pieces(i, p);
  }

  // Per instance: aA = (start xyz, radius µm), aB = (end xyz, radius µm), aC = (start path µm,
  // end path µm, neuron, compartment), aTA / aTB = the curve tangent at each end. Pieces that
  // meet share a tangent, so their quads join edge to edge with no wedge-shaped gap at a bend.
  const a = new Float32Array(edges * 4);
  const b = new Float32Array(edges * 4);
  const cc = new Float32Array(edges * 4);
  const ta = new Float32Array(edges * 3);
  const tb = new Float32Array(edges * 3);
  const pt = [0, 0, 0];
  const tg = [0, 0, 0];
  const mA = [0, 0, 0];
  const mB = [0, 0, 0];
  // Point and unit tangent at parameter s along the Hermite curve of the current edge.
  const hermite = (i: number, p: number, s: number) => {
    const s2 = s * s;
    const s3 = s2 * s;
    const h00 = 2 * s3 - 3 * s2 + 1;
    const h10 = s3 - 2 * s2 + s;
    const h01 = -2 * s3 + 3 * s2;
    const h11 = s3 - s2;
    const d00 = 6 * s2 - 6 * s;
    const d10 = 3 * s2 - 4 * s + 1;
    const d01 = -6 * s2 + 6 * s;
    const d11 = 3 * s2 - 2 * s;
    for (let k = 0; k < 3; k++) {
      const pa = pos[i * 3 + k];
      const pb = pos[p * 3 + k];
      pt[k] = h00 * pa + h10 * mA[k] + h01 * pb + h11 * mB[k];
      tg[k] = d00 * pa + d10 * mA[k] + d01 * pb + d11 * mB[k];
    }
    const l = Math.hypot(tg[0], tg[1], tg[2]) || 1;
    tg[0] /= l;
    tg[1] /= l;
    tg[2] /= l;
  };

  let e = 0;
  for (let i = 0; i < nodes.count; i++) {
    const p = nodes.parent[i];
    if (p === NO_PARENT) continue;
    const rChild = nodes.radiusNm[i] / 1000;
    let rParent = nodes.radiusNm[p] / 1000;
    // An edge that leaves the soma would otherwise start as wide as the whole cell body.
    if (nodes.compartment[p] === 0) rParent = Math.min(rChild * 2, rParent * 0.3);
    const pathChild = nodes.pathDistQ[i] * PATH_DIST_UNIT_UM;
    const pathParent = nodes.pathDistQ[p] * PATH_DIST_UNIT_UM;
    const neuron = nodes.neuronOfNode[i];
    const compartment = nodes.compartment[i];

    const len = Math.hypot(pos[p * 3] - pos[i * 3], pos[p * 3 + 1] - pos[i * 3 + 1], pos[p * 3 + 2] - pos[i * 3 + 2]);
    // Only the parent's continuing child shares its tangent; side branches leave straight.
    const tParent = mainChild[p] === i && nodes.compartment[p] !== 0 ? p : i;
    for (let k = 0; k < 3; k++) {
      mA[k] = tangents[i * 3 + k] * len;
      mB[k] = tangents[tParent * 3 + k] * len;
    }

    const n = pieces(i, p);
    hermite(i, p, 0);
    let [x0, y0, z0] = pt;
    let [tx0, ty0, tz0] = tg;
    for (let j = 0; j < n; j++) {
      const s0 = j / n;
      const s1 = (j + 1) / n;
      hermite(i, p, s1);
      a.set([x0, y0, z0, rChild + (rParent - rChild) * s0], e * 4);
      b.set([pt[0], pt[1], pt[2], rChild + (rParent - rChild) * s1], e * 4);
      cc.set([pathChild + (pathParent - pathChild) * s0, pathChild + (pathParent - pathChild) * s1, neuron, compartment], e * 4);
      ta.set([tx0, ty0, tz0], e * 3);
      tb.set([tg[0], tg[1], tg[2]], e * 3);
      [x0, y0, z0] = pt;
      [tx0, ty0, tz0] = tg;
      e++;
    }
  }

  const geometry = new InstancedBufferGeometry();
  // Corner x across the ribbon in [-1, 1], y along it in [0, 1] (0 at the child).
  geometry.setAttribute("position", new Float32BufferAttribute([-1, 0, 0, 1, 0, 0, 1, 1, 0, -1, 1, 0], 3));
  geometry.setIndex([0, 1, 2, 0, 2, 3]);
  geometry.setAttribute("aA", new InstancedBufferAttribute(a, 4));
  geometry.setAttribute("aB", new InstancedBufferAttribute(b, 4));
  geometry.setAttribute("aC", new InstancedBufferAttribute(cc, 4));
  geometry.setAttribute("aTA", new InstancedBufferAttribute(ta, 3));
  geometry.setAttribute("aTB", new InstancedBufferAttribute(tb, 3));
  geometry.instanceCount = edges;

  const ribbons = new Mesh(geometry, ribbonMaterial(u, textures, slots));
  ribbons.frustumCulled = false;

  // Somas.
  const somaCount = data.neurons.count;
  const somaPos = new Float32Array(somaCount * 3);
  const somas = new InstancedMesh(new SphereGeometry(1, 32, 20), somaMaterial(u, textures, slots), somaCount);
  const m = new Matrix4();
  const v = new Vector3();
  for (let n = 0; n < somaCount; n++) {
    const s = data.neuronNodeStart[n];
    v.set(pos[s * 3], pos[s * 3 + 1], pos[s * 3 + 2]);
    somaPos.set([v.x, v.y, v.z], n * 3);
    const radius = Math.max(2.5, Math.min(4.5, nodes.radiusNm[s] / 1000));
    m.makeScale(radius, radius, radius).setPosition(v);
    somas.setMatrixAt(n, m);
  }
  somas.instanceMatrix.needsUpdate = true;
  somas.frustumCulled = false;

  const pathTex = axonPathTexture(data, pos);
  const pulses = pulseSprites(u, slots, pathTex, data.neurons.count);
  const synapseGlow = opts.synapses?.count ? synapseSprites(u, slots, opts.synapses, bmin, bmax) : null;

  const source = opts.spikes;
  let uploaded = -1;
  return {
    ribbons,
    somas,
    pulses,
    synapseGlow,
    textures,
    instanceCount: edges,
    somaPos,
    update() {
      if (source?.kind !== "array" || source.version === uploaded) return;
      uploaded = source.version;
      textures.spikeTimes.set(source.times);
      textures.spikes.needsUpdate = true;
    },
    dispose() {
      for (const m of [pulses, synapseGlow]) {
        if (!m) continue;
        m.geometry.dispose();
        (m.material as MeshBasicNodeMaterial).dispose();
      }
      pathTex.dispose();
      geometry.dispose();
      (ribbons.material as MeshBasicNodeMaterial).dispose();
      somas.geometry.dispose();
      (somas.material as MeshBasicNodeMaterial).dispose();
      somas.dispose();
      textures.info.dispose();
      textures.spikes.dispose();
    },
  };
}

/** Exponential haze with view depth: 1 up close, almost nothing at the far side of the volume. */
function haze(u: SceneUniforms, depth: FloatNode) {
  return exp(depth.div(u.hazeDistance).negate());
}

function ribbonMaterial(u: SceneUniforms, tex: NeuronTextures, slots: SpikeSlots): MeshBasicNodeMaterial {
  const aA = attribute("aA", "vec4");
  const aB = attribute("aB", "vec4");
  const aC = attribute("aC", "vec4");
  const along = positionLocal.y;
  const across = positionLocal.x;

  const pA = modelViewMatrix.mul(vec4(aA.xyz, 1)).xyz;
  const pB = modelViewMatrix.mul(vec4(aB.xyz, 1)).xyz;
  const p = mix(pA, pB, along);
  const aTA = attribute("aTA", "vec3");
  const aTB = attribute("aTB", "vec3");
  const tangent = modelViewMatrix.mul(vec4(mix(aTA, aTB, along), 0)).xyz;
  const side0 = cross(tangent, p.negate());
  const sideLen = length(side0);
  const side = select(sideLen.greaterThan(1e-6), side0.div(sideLen), vec3(1, 0, 0));

  const depth = max(p.z.negate(), 0.5);
  // Pixels per µm at this depth: proj[1][1] * half the drawing-buffer height / depth.
  const pxPerUm = u.pixelScale.div(depth);
  const radius = max(mix(aA.w, aB.w, along), u.radiusFloor);
  const radiusPx = radius.mul(pxPerUm);
  const drawnPx = max(radiusPx, u.minHalfWidthPx);
  const halfWidth = drawnPx.div(pxPerUm);
  const posView = p.add(side.mul(across).mul(halfWidth));

  const material = new MeshBasicNodeMaterial();
  material.vertexNode = cameraProjectionMatrix.mul(vec4(posView, 1));

  const neuron = aC.z;
  const info = textureLoad(tex.info, texel(neuron, 1, 0));
  const compartment = aC.w;
  const isAxon = step(0.5, compartment).mul(step(compartment, 1.5));
  const path = mix(aC.x, aC.y, along);

  // Everything that is constant or smooth along one piece is worked out per vertex and
  // interpolated; the fragment stage only shapes the cross-section and the travelling pulse.
  const h = haze(u, depth);
  const coverage = min(radiusPx.div(drawnPx), 1);
  // Far structure drifts toward the haze tint, near structure keeps its own hue.
  const tint = mix(u.hazeTint, info.rgb, h.pow(0.35));
  const vBase = varying(tint.mul(info.a).mul(mix(float(1), u.axonGain, isAxon)).mul(u.exposure));
  const vAlpha = varying(
    u.opacity.mul(coverage).mul(h).mul(mix(float(1), u.axonAlpha, isAxon)).mul(sqrt(info.a)),
  );
  // Back-propagating spikes reach into the dendrites weakly and die out with distance.
  const glowGain = mix(exp(path.negate().div(90)).mul(0.35), float(1), isAxon);
  const vGlow = varying(glowGain.mul(spikeNear(u, depth, neuron)).mul(sqrt(coverage)).mul(h));
  const vDrawnPx = varying(drawnPx);
  const vDepth = varying(depth);
  const delay = path.mul(u.slowMo).mul(1e-6).div(u.velocity);
  const vDelay = varying(delay);
  // Of the neuron's 8 spike slots, only the latest one whose front has reached (or is about to
  // reach) this piece lights it: an earlier spike's pulse has passed and its tail is faint by then.
  // Picked once per vertex, so each fragment evaluates one pulse instead of eight.
  const midDelay = mix(aC.x, aC.y, 0.5).mul(u.slowMo).mul(1e-6).div(u.velocity);
  const [slotA, slotB] = slots(neuron);
  const horizon = u.time.sub(midDelay).add(u.pulseWidth.mul(3));
  let latest: FloatNode = float(NEVER);
  for (const t of [slotA.x, slotA.y, slotA.z, slotA.w, slotB.x, slotB.y, slotB.z, slotB.w]) {
    latest = select(t.lessThanEqual(horizon).and(t.greaterThan(latest)), t, latest);
  }
  const vSpike = varying(latest);
  const vAcross = varying(across);
  const vThick = varying(smoothstep(2.5, 14, radiusPx));

  // Thin ribbons (a few px) get a soft bright core. Thick ones, which only happen close to the
  // camera, read as translucent tubes instead: a dim body with brighter membrane edges.
  const x2 = vAcross.mul(vAcross);
  const facing = sqrt(max(float(1).sub(x2), 0));
  const core = pow(facing, 3.2);
  const rim = float(1).sub(facing);
  const membrane = float(0.16).add(rim.mul(rim).mul(0.9)).mul(smoothstep(1, 0.82, abs(vAcross)));
  const profile = mix(core, membrane.mul(0.7), vThick);

  const dt = u.time.sub(vSpike.add(vDelay));
  const x = dt.div(u.pulseWidth);
  const pulse = exp(x.mul(x).negate());
  const after = step(0, dt).mul(exp(dt.negate().div(u.afterglow)));
  const base = vBase;
  // Anything within a few tens of µm of the lens falls away, as if out of the focal plane. Kept
  // per fragment: it changes steeply along the long pieces that pass right by the camera.
  const nearFade = smoothstep(4, 45, vDepth);
  // Premultiplied "over" blending: each ribbon partly hides what is behind it, so dense tangles
  // settle toward the ribbons' own colour instead of summing to white. The spike glow rides on
  // top as pure emission (colour beyond alpha), which is what bloom picks up.
  const alpha = vAlpha.mul(profile).mul(nearFade);
  // The glowing core is the thin ribbon's own core, but never wider than glowCapPx: a thick ribbon
  // up close carries its pulse as a bright line inside the tube.
  const t = vAcross.mul(max(float(1), vDrawnPx.div(u.glowCapPx)));
  const glowCore = pow(sqrt(max(float(1).sub(t.mul(t)), 0)), 3.2);
  const emission = u.spikeColor
    .mul(pulse.mul(u.spikeGain).add(after.mul(u.afterglowGain)))
    .mul(vGlow.mul(nearFade))
    .mul(glowCore);

  material.colorNode = vec4(base.mul(alpha).add(emission), alpha);
  material.transparent = true;
  material.depthWrite = false;
  material.blending = CustomBlending;
  material.blendSrc = OneFactor;
  material.blendDst = OneMinusSrcAlphaFactor;
  material.blendSrcAlpha = OneFactor;
  material.blendDstAlpha = OneMinusSrcAlphaFactor;
  return material;
}

function somaMaterial(u: SceneUniforms, tex: NeuronTextures, slots: SpikeSlots): MeshBasicNodeMaterial {
  const neuron = instanceIndex.toFloat();
  const info = varying(textureLoad(tex.info, texel(neuron, 1, 0)));
  const [slotA, slotB] = slots(neuron);
  const spikeA = varying(slotA);
  const spikeB = varying(slotB);

  const depth = positionView.z.negate();
  const facing = abs(normalView.dot(positionView.normalize().negate()));
  // An emissive translucent body: brightness follows the path length through the sphere, so
  // the centre is densest, with a faint membrane rim.
  const warm = mix(info.rgb, vec3(1.0, 0.85, 0.7), 0.12);
  const body = warm
    .mul(float(0.55).add(info.a.mul(0.45)))
    .mul(pow(facing, 3).mul(0.5).add(pow(float(1).sub(facing), 3).mul(0.12)))
    .mul(u.somaGain);
  const { pulse, after } = spikeGlow(u, spikeA, spikeB, float(0), u.somaAfterglow);
  // A soma is 2.5 to 4.5 µm across; once it is larger than glowCapPx on screen its flash keeps a
  // constant total light instead of growing with its area.
  const radiusPx = u.pixelScale.mul(SOMA_GLOW_UM).div(max(depth, 0.5));
  const cap = min(float(1), u.glowCapPx.div(radiusPx)).pow(2);
  const glow = u.spikeColor
    .mul(pulse.mul(u.spikeGain).add(after.mul(u.somaAfterglowGain)))
    .mul(float(0.3).add(facing.mul(0.7)))
    .mul(spikeNear(u, depth, neuron))
    .mul(cap);

  const material = new MeshBasicNodeMaterial();
  material.colorNode = vec4(body.add(glow).mul(haze(u, depth)).mul(smoothstep(8, 60, depth)), 1);
  material.transparent = true;
  material.depthWrite = false;
  material.blending = AdditiveBlending;
  return material;
}

/**
 * Each neuron's longest axon branch (soma to its farthest axon tip), resampled at PATH_SAMPLES
 * points evenly spaced in path length: rgb = position, a = the branch's total length in µm.
 * A neuron with no axon keeps length 0 and never shows a pulse sprite.
 */
function axonPathTexture(data: Dataset, pos: Float32Array): DataTexture {
  const { nodes } = data;
  const n = data.neurons.count;
  const tex = makeTexture(n * PATH_SAMPLES);
  const out = tex.image.data as Float32Array;
  const pathUm = (i: number) => nodes.pathDistQ[i] * PATH_DIST_UNIT_UM;
  const chain: number[] = [];
  for (let k = 0; k < n; k++) {
    const start = data.neuronNodeStart[k];
    const end = start + data.neuronNodeCount[k];
    let tip = -1;
    for (let i = start; i < end; i++) {
      if (nodes.compartment[i] === 1 && (tip < 0 || nodes.pathDistQ[i] > nodes.pathDistQ[tip])) tip = i;
    }
    if (tip < 0) continue;
    chain.length = 0;
    for (let i = tip; i !== NO_PARENT && chain.length <= end - start; i = nodes.parent[i]) chain.push(i);
    chain.reverse();
    const len = pathUm(tip);
    let j = 0;
    for (let s = 0; s < PATH_SAMPLES; s++) {
      const d = (s / (PATH_SAMPLES - 1)) * len;
      while (j < chain.length - 2 && pathUm(chain[j + 1]) < d) j++;
      const a = chain[j];
      const b = chain[Math.min(j + 1, chain.length - 1)];
      const span = pathUm(b) - pathUm(a);
      const f = span > 0 ? Math.min(1, Math.max(0, (d - pathUm(a)) / span)) : 0;
      const o = (k * PATH_SAMPLES + s) * 4;
      for (let c = 0; c < 3; c++) out[o + c] = pos[a * 3 + c] + (pos[b * 3 + c] - pos[a * 3 + c]) * f;
      out[o + 3] = len;
    }
  }
  return tex;
}

/** A unit quad, corners in [-1, 1], drawn `count` times. */
function quadGeometry(count: number): InstancedBufferGeometry {
  const geometry = new InstancedBufferGeometry();
  geometry.setAttribute("position", new Float32BufferAttribute([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0], 3));
  geometry.setIndex([0, 1, 2, 0, 2, 3]);
  geometry.instanceCount = count;
  return geometry;
}

/** Slot `slot` (0..7, a uint node) of a neuron's two spike vec4s. */
function pickSlot([a, b]: SlotPair, slot: Node<"uint">): FloatNode {
  const parts = [a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w];
  let v: FloatNode = parts[0];
  for (let k = 1; k < parts.length; k++) v = select(slot.equal(uint(k)), parts[k], v);
  return v;
}

/**
 * Additive camera-facing point lights. Sprites whose light is negligible collapse to zero size
 * in the vertex stage, so the idle majority costs no fragments.
 */
function spriteMesh(
  u: SceneUniforms,
  geometry: InstancedBufferGeometry,
  neuron: FloatNode,
  centre: Node<"vec3">,
  intensity: FloatNode,
  radiusUm: FloatNode,
  color: Node<"vec3">,
): Mesh {
  const centreView = modelViewMatrix.mul(vec4(centre, 1)).xyz;
  const depth = max(centreView.z.negate(), 0.5);
  const pxPerUm = u.pixelScale.div(depth);
  // At least 2.5 px so a far point still reads, at most glowCapPx so a near one is not a blob.
  const radius = min(max(radiusUm, float(2.5).div(pxPerUm)), u.glowCapPx.div(pxPerUm));
  const light = intensity.mul(spikeNear(u, depth, neuron)).mul(haze(u, depth)).mul(smoothstep(4, 45, depth));
  const size = select(light.greaterThan(0.002), radius, float(0));
  const corner = positionLocal.xy;

  const material = new MeshBasicNodeMaterial();
  material.vertexNode = cameraProjectionMatrix.mul(vec4(centreView.add(vec3(corner.mul(size), 0)), 1));
  const vCorner = varying(corner);
  const vLight = varying(light);
  const falloff = exp(vCorner.dot(vCorner).mul(-4));
  material.colorNode = vec4(color.mul(vLight).mul(falloff), 1);
  material.transparent = true;
  material.depthWrite = false;
  material.blending = AdditiveBlending;

  const mesh = new Mesh(geometry, material);
  mesh.frustumCulled = false;
  return mesh;
}

/** One sprite per (neuron, spike slot), riding that spike's front along the longest axon branch. */
function pulseSprites(u: SceneUniforms, slots: SpikeSlots, pathTex: DataTexture, neuronCount: number): Mesh {
  const neuron = instanceIndex.div(SPIKE_SLOTS).toFloat();
  const tau = pickSlot(slots(neuron), instanceIndex.bitAnd(SPIKE_SLOTS - 1));
  const len = textureLoad(pathTex, texel(neuron, PATH_SAMPLES, 0)).w;
  // Inverse of the ribbon shader's delay: how far along the path the front is now, µm.
  const d = u.time.sub(tau).mul(u.velocity).div(u.slowMo.mul(1e-6));
  const f = clamp(d.div(max(len, 1e-3)), 0, 1).mul(PATH_SAMPLES - 1);
  const i0 = floor(f);
  const i1 = min(i0.add(1), PATH_SAMPLES - 1);
  const p = mix(
    textureLoad(pathTex, texel(neuron, PATH_SAMPLES, i0)).xyz,
    textureLoad(pathTex, texel(neuron, PATH_SAMPLES, i1)).xyz,
    f.sub(i0),
  );
  const alive = step(0, d)
    .mul(step(d, len))
    .mul(smoothstep(0, 12, d))
    .mul(float(1).sub(smoothstep(len.sub(25), len, d)));
  const color = mix(u.spikeColor, vec3(1, 0.95, 0.85), 0.5);
  return spriteMesh(u, quadGeometry(neuronCount * SPIKE_SLOTS), neuron, p, alive.mul(u.pulseSpriteGain), u.pulseSpriteUm, color);
}

/** One sprite per synapse, lit briefly each time a pulse from its presynaptic cell lands there. */
function synapseSprites(
  u: SceneUniforms,
  slots: SpikeSlots,
  syn: SynapseTable,
  bmin: [number, number, number],
  bmax: [number, number, number],
): Mesh {
  const geometry = quadGeometry(syn.count);
  const at = new Float32Array(syn.count * 4);
  const pre = new Float32Array(syn.count);
  for (let s = 0; s < syn.count; s++) {
    for (let k = 0; k < 3; k++) at[s * 4 + k] = dequantise(syn.pos[s * 3 + k], bmin[k], bmax[k]);
    at[s * 4 + 3] = syn.preDistQ[s] * PATH_DIST_UNIT_UM;
    pre[s] = syn.pre[s];
  }
  geometry.setAttribute("aSyn", new InstancedBufferAttribute(at, 4));
  geometry.setAttribute("aPre", new InstancedBufferAttribute(pre, 1));
  const aSyn = attribute("aSyn", "vec4");
  const aPre = attribute("aPre", "float");

  const [a, b] = slots(aPre);
  const delay = aSyn.w.mul(u.slowMo).mul(1e-6).div(u.velocity).add(u.synDelay);
  let glow: FloatNode = float(0);
  for (const s of [a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w]) {
    const dt = u.time.sub(s.add(delay));
    // A 20 ms rise so it does not pop, then an exponential fade.
    glow = glow.add(smoothstep(-0.02, 0, dt).mul(exp(max(dt, 0).negate().div(u.synapseGlowDecay))));
  }
  const color = mix(u.spikeColor, vec3(1, 0.92, 0.8), 0.35);
  return spriteMesh(u, geometry, aPre, aSyn.xyz, glow.mul(u.synapseGlowGain), u.synapseGlowUm, color);
}
