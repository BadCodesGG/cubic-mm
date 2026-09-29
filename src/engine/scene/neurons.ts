/**
 * The neurons: every skeleton edge (child to parent) is one instance of a camera-facing ribbon,
 * and every soma one instance of a sphere. Both are additive light, so the volume reads as a
 * haze of faint structure that brightens only where it is dense, near, or carrying a spike.
 *
 * Per-neuron data lives in two float textures indexed by neuron number, read in the vertex
 * stage with `textureLoad`, which works identically on the WebGPU and WebGL2 backends:
 *   info:   one texel per neuron, rgb = base tint (linear), a = brightness
 *   spikes: two texels per neuron, the last 8 spike times in seconds (ring buffer)
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
  textureLoad,
  varying,
  vec3,
  vec4,
} from "three/tsl";
import type { Node } from "three/webgpu";
import { NO_PARENT, PATH_DIST_UNIT_UM, dequantise } from "../format";
import type { Dataset } from "../data";
import { rng } from "../synth";
import type { SceneUniforms } from "./uniforms";

export const SPIKE_SLOTS = 8;
/** Spike slots start here: long enough ago that nothing glows. */
export const NEVER = -1e4;
const TEX_WIDTH = 1024;
/** Longest piece a skeleton edge is drawn with, µm. */
const PIECE_UM = 3;

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
function texel(index: FloatNode, stride: number, k: number) {
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
export function createNeurons(data: Dataset, u: SceneUniforms, maxPieces = 3): Neurons {
  const { nodes, manifest } = data;
  const { min: bmin, max: bmax } = manifest.boundsUm;
  const textures = createNeuronTextures(data);

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

  const ribbons = new Mesh(geometry, ribbonMaterial(u, textures));
  ribbons.frustumCulled = false;

  // Somas.
  const somaCount = data.neurons.count;
  const somaPos = new Float32Array(somaCount * 3);
  const somas = new InstancedMesh(new SphereGeometry(1, 32, 20), somaMaterial(u, textures), somaCount);
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

  return {
    ribbons,
    somas,
    textures,
    instanceCount: edges,
    somaPos,
    dispose() {
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

function ribbonMaterial(u: SceneUniforms, tex: NeuronTextures): MeshBasicNodeMaterial {
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
  const vAcross = varying(across);
  const vPath = varying(mix(aC.x, aC.y, along));
  const vCoverage = varying(min(radiusPx.div(drawnPx), 1));
  const vDepth = varying(depth);
  const vCompartment = varying(aC.w);
  const vInfo = varying(textureLoad(tex.info, texel(neuron, 1, 0)));
  const vSpikeA = varying(textureLoad(tex.spikes, texel(neuron, 2, 0)));
  const vSpikeB = varying(textureLoad(tex.spikes, texel(neuron, 2, 1)));

  const vRadiusPx = varying(radiusPx);
  const isAxon = step(0.5, vCompartment).mul(step(vCompartment, 1.5));

  // Thin ribbons (a few px) get a soft bright core. Thick ones, which only happen close to the
  // camera, read as translucent tubes instead: a dim body with brighter membrane edges.
  const x2 = vAcross.mul(vAcross);
  const facing = sqrt(max(float(1).sub(x2), 0));
  const core = pow(facing, 3.2);
  const membrane = float(0.16).add(pow(float(1).sub(facing), 2).mul(0.9)).mul(smoothstep(1, 0.82, abs(vAcross)));
  const thick = smoothstep(2.5, 14, vRadiusPx);
  const profile = mix(core, membrane.mul(0.7), thick);

  const h = haze(u, vDepth);
  // Anything within a few tens of µm of the lens falls away, as if out of the focal plane.
  const nearFade = smoothstep(4, 45, vDepth);
  // Far structure drifts toward the haze tint, near structure keeps its own hue.
  const tint = mix(u.hazeTint, vInfo.rgb, h.pow(0.35));
  const compGain = mix(float(1), u.axonGain, isAxon);
  const base = tint.mul(vInfo.a).mul(compGain).mul(u.exposure);

  const { pulse, after } = spikeGlow(u, vSpikeA, vSpikeB, vPath);
  // Back-propagating spikes reach into the dendrites weakly and die out with distance.
  const glowGain = mix(exp(vPath.negate().div(90)).mul(0.35), float(1), isAxon);
  const glow = u.spikeColor.mul(pulse.mul(u.spikeGain).add(after.mul(u.afterglowGain))).mul(glowGain);
  const fade = h.mul(nearFade);
  // Premultiplied "over" blending: each ribbon partly hides what is behind it, so dense tangles
  // settle toward the ribbons' own colour instead of summing to white. The spike glow rides on
  // top as pure emission (colour beyond alpha), which is what bloom picks up.
  const alpha = u.opacity
    .mul(vCoverage)
    .mul(profile)
    .mul(fade)
    .mul(mix(float(1), u.axonAlpha, isAxon))
    .mul(sqrt(vInfo.a));
  const emission = glow.mul(sqrt(vCoverage)).mul(core).mul(fade);

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

function somaMaterial(u: SceneUniforms, tex: NeuronTextures): MeshBasicNodeMaterial {
  const neuron = instanceIndex.toFloat();
  const info = varying(textureLoad(tex.info, texel(neuron, 1, 0)));
  const spikeA = varying(textureLoad(tex.spikes, texel(neuron, 2, 0)));
  const spikeB = varying(textureLoad(tex.spikes, texel(neuron, 2, 1)));

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
  const glow = u.spikeColor
    .mul(pulse.mul(u.spikeGain).add(after.mul(u.somaAfterglowGain)))
    .mul(float(0.3).add(facing.mul(0.7)));

  const material = new MeshBasicNodeMaterial();
  material.colorNode = vec4(body.add(glow).mul(haze(u, depth)).mul(smoothstep(8, 60, depth)), 1);
  material.transparent = true;
  material.depthWrite = false;
  material.blending = AdditiveBlending;
  return material;
}
