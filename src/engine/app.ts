/**
 * Wires the experience together: renderer, dataset, scene, camera, post, a round-0 fake spike
 * schedule, and the animation loop. `startApp` resolves once the first frame is ready to draw.
 *
 * `?shot=hero` runs a fixed-timestep clock and the scripted hero camera, and holds the
 * simulation at a fixed time so screenshots are reproducible.
 */

import { PerspectiveCamera, Scene, Vector3 } from "three/webgpu";
import { NO_PARENT, PATH_DIST_UNIT_UM, dequantise } from "./format";
import { loadDataset, type Dataset } from "./data";
import { rng, synthDataset } from "./synth";
import { createRenderer } from "./scene/renderer";
import { createNeurons, SPIKE_SLOTS, type Neurons } from "./scene/neurons";
import { createLayers } from "./scene/layers";
import { createPost } from "./scene/post";
import { createSceneUniforms } from "./scene/uniforms";
import { FlyControls, heroPath, WORLD_UP, type CameraPose, type HeroAnchor } from "./camera/fly";

export interface AppOptions {
  /** "hero" selects the scripted, reproducible camera. */
  shot?: string | null;
  /** Use the synthetic dataset instead of /data. */
  synth?: boolean;
  forceWebGL?: boolean;
  /** Simulation time the hero shot holds at, seconds. */
  holdAt?: number;
  onProgress?: (fraction: number, label: string) => void;
}

export interface App {
  isWebGPU: boolean;
  dispose(): void;
}

export interface CmmDebug {
  isWebGPU: boolean;
  frame: number;
  spikes: number;
  instances: number;
  /** Mean wall-clock ms between the last 60 frames. */
  frameMs: number;
  simTime: number;
  /** True once a shot's clock has reached its hold time. */
  settled: boolean;
  dataset: "real" | "synth";
  hero: { neuron: number; distanceUm: number; screen: [number, number] };
}

declare global {
  interface Window {
    __cmm?: CmmDebug;
  }
}

const HERO_INTERVAL = 2;
const RANDOM_MEAN_INTERVAL = 20;

/** Round-0 stand-in for the simulation: Poisson spikes everywhere, plus a metronome neuron. */
class SpikeSchedule {
  private readonly next: Float64Array;
  private readonly slot: Uint8Array;
  private readonly r: () => number;
  heroNext: number;
  count = 0;
  dirty = true;

  constructor(
    private readonly times: Float32Array,
    neuronCount: number,
    seed: number,
    firstHeroSpike: number,
  ) {
    this.r = rng(seed);
    this.next = new Float64Array(neuronCount);
    this.slot = new Uint8Array(neuronCount);
    this.heroNext = firstHeroSpike;
    // Start from 20 s of history so the volume already has afterglow in it at t = 0.
    for (let n = 0; n < neuronCount; n++) {
      let t = -RANDOM_MEAN_INTERVAL + this.interval();
      while (t < 0) {
        this.emit(n, t);
        t += this.interval();
      }
      this.next[n] = t;
    }
    this.count = 0;
  }

  private interval(): number {
    return -Math.log(1 - this.r()) * RANDOM_MEAN_INTERVAL;
  }

  private emit(n: number, t: number): void {
    this.times[n * SPIKE_SLOTS + this.slot[n]] = t;
    this.slot[n] = (this.slot[n] + 1) % SPIKE_SLOTS;
    this.count++;
    this.dirty = true;
  }

  advance(t: number, hero: () => number): void {
    while (t >= this.heroNext) {
      this.emit(hero(), this.heroNext);
      this.heroNext += HERO_INTERVAL;
    }
    for (let n = 0; n < this.next.length; n++) {
      while (t >= this.next[n]) {
        this.emit(n, this.next[n]);
        this.next[n] += this.interval();
      }
    }
  }
}

/**
 * The hero: a neuron on the edge of the soma cluster whose axon runs near-horizontally back in
 * toward the cluster. The camera sits behind it looking along that axon, so the frame holds
 * the near soma, its axon carrying the pulse, and the rest of the cells as depth behind.
 * Excitatory cells are preferred.
 */
function pickHero(data: Dataset): { neuron: number; anchor: HeroAnchor } {
  const { neurons, nodes, manifest } = data;
  const { min: bmin, max: bmax } = manifest.boundsUm;
  const med = (k: number) => {
    const v = Array.from({ length: neurons.count }, (_, i) => neurons.somaUm[i * 3 + k]).sort((a, b) => a - b);
    return v[v.length >> 1];
  };
  const centre = new Vector3(med(0), med(1), med(2));
  // Median soma distance from the centre: "the edge of the cluster" in this dataset's own scale.
  const radii = Array.from({ length: neurons.count }, (_, i) =>
    new Vector3().fromArray(neurons.somaUm, i * 3).distanceTo(centre),
  ).sort((a, b) => a - b);
  const edge = radii[Math.floor(radii.length * 0.7)];
  const node = (i: number) =>
    new Vector3(
      dequantise(nodes.pos[i * 3], bmin[0], bmax[0]),
      dequantise(nodes.pos[i * 3 + 1], bmin[1], bmax[1]),
      dequantise(nodes.pos[i * 3 + 2], bmin[2], bmax[2]),
    );

  let best = 0;
  let bestScore = Infinity;
  let bestAxis = new Vector3(1, 0, 0);
  for (let n = 0; n < neurons.count; n++) {
    const start = data.neuronNodeStart[n];
    const end = start + data.neuronNodeCount[n];
    // The axon node closest to 200 µm of path from the soma.
    let axonNode = -1;
    let axonErr = Infinity;
    for (let i = start; i < end; i++) {
      if (nodes.compartment[i] !== 1 || nodes.parent[i] === NO_PARENT) continue;
      const err = Math.abs(nodes.pathDistQ[i] * PATH_DIST_UNIT_UM - 200);
      if (err < axonErr) {
        axonErr = err;
        axonNode = i;
      }
    }
    if (axonNode < 0 || axonErr > 60) continue;
    const soma = new Vector3().fromArray(neurons.somaUm, n * 3);
    const axis = node(axonNode).sub(soma);
    if (axis.lengthSq() < 100) continue;
    axis.normalize();
    const inward = centre.clone().sub(soma).normalize().dot(axis);
    const score =
      Math.abs(soma.distanceTo(centre) - edge) +
      300 * Math.abs(axis.y) +
      150 * (1 - inward) +
      (neurons.inhibitory[n] ? 150 : 0);
    if (score < bestScore) {
      bestScore = score;
      best = n;
      bestAxis = axis;
    }
  }
  const soma = new Vector3().fromArray(neurons.somaUm, best * 3);
  // Keep the view from pointing straight down the depth axis, where "up" is undefined.
  const axis = bestAxis.clone();
  axis.y *= 0.6;
  axis.normalize();
  return { neuron: best, anchor: { soma, axis } };
}

function nearestSoma(somaPos: Float32Array, p: Vector3): number {
  let best = 0;
  let bestD = Infinity;
  for (let n = 0; n < somaPos.length / 3; n++) {
    const dx = somaPos[n * 3] - p.x;
    const dy = somaPos[n * 3 + 1] - p.y;
    const dz = somaPos[n * 3 + 2] - p.z;
    const d = dx * dx + dy * dy + dz * dz;
    if (d < bestD) {
      bestD = d;
      best = n;
    }
  }
  return best;
}

export async function startApp(canvas: HTMLCanvasElement, opts: AppOptions = {}): Promise<App> {
  const progress = opts.onProgress ?? (() => {});
  const shot = opts.shot === "hero";
  const abort = new AbortController();

  progress(0.02, "Starting the renderer");
  const { renderer, isWebGPU } = await createRenderer(canvas, { forceWebGL: opts.forceWebGL });

  let data: Dataset;
  let source: CmmDebug["dataset"] = "real";
  if (opts.synth) {
    progress(0.1, "Growing synthetic neurons");
    await new Promise((r) => setTimeout(r, 0));
    data = synthDataset(1, 200);
    source = "synth";
  } else {
    try {
      data = await loadDataset(
        "/data",
        isWebGPU ? "hi" : "lite",
        (done, total, label) => progress(0.05 + 0.8 * (done / Math.max(1, total)), `Loading ${label}`),
        abort.signal,
      );
    } catch (err) {
      console.warn("Real dataset unavailable, using the synthetic stand-in:", err);
      data = synthDataset(1, 200);
      source = "synth";
    }
  }

  progress(0.9, "Building the volume");
  const u = createSceneUniforms();
  const scene = new Scene();
  const neurons: Neurons = createNeurons(data, u, isWebGPU ? 3 : 2);
  scene.add(neurons.ribbons, neurons.somas);
  const layers = createLayers(data, u);
  scene.add(layers.group);

  const camera = new PerspectiveCamera(55, 1, 0.5, 4000);
  camera.up.copy(WORLD_UP);
  const hero = pickHero(data);
  const pose: CameraPose = { position: new Vector3(), target: new Vector3() };
  heroPath(0, hero.anchor, pose);
  camera.position.copy(pose.position);
  camera.lookAt(pose.target);

  const controls = shot ? null : new FlyControls(camera, canvas);
  controls?.lookAt(pose.target);

  const post = createPost(renderer, scene, camera);

  const holdAt = opts.holdAt ?? 1.3;
  const schedule = new SpikeSchedule(neurons.textures.spikeTimes, data.neurons.count, 11, shot ? 1.0 : 0.6);
  let heroNeuron = hero.neuron;
  const pickHeroNeuron = () => {
    if (!shot) heroNeuron = nearestSoma(neurons.somaPos, camera.position);
    return heroNeuron;
  };

  const debug: CmmDebug = {
    isWebGPU,
    frame: 0,
    spikes: 0,
    instances: neurons.instanceCount,
    frameMs: 0,
    simTime: 0,
    settled: false,
    dataset: source,
    hero: { neuron: hero.neuron, distanceUm: 0, screen: [0, 0] },
  };
  window.__cmm = debug;

  const resize = () => {
    const w = canvas.clientWidth || window.innerWidth;
    const h = canvas.clientHeight || window.innerHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    const bufferHeight = h * renderer.getPixelRatio();
    u.pixelScale.value = camera.projectionMatrix.elements[5] * 0.5 * bufferHeight;
    u.minHalfWidthPx.value = 0.8 * renderer.getPixelRatio();
  };
  resize();
  const observer = new ResizeObserver(resize);
  observer.observe(canvas);

  let simTime = 0;
  let last = -1;
  const deltas: number[] = [];
  let disposed = false;
  const loop = () => {
    // React strict mode mounts twice in dev and disposes the first app as soon as it resolves; the
    // renderer's loop can still fire once after that, on a detached zero-size canvas.
    if (disposed || !canvas.isConnected || canvas.clientWidth === 0 || canvas.clientHeight === 0) return;
    const now = performance.now();
    const dt = last < 0 ? 1 / 60 : Math.min(0.1, (now - last) / 1000);
    if (last >= 0) {
      deltas.push(now - last);
      if (deltas.length > 60) deltas.shift();
      debug.frameMs = deltas.reduce((s, d) => s + d, 0) / deltas.length;
    }
    last = now;

    if (shot) {
      simTime = Math.min(debug.frame / 60, holdAt);
      debug.settled = simTime >= holdAt;
      heroPath(simTime, hero.anchor, pose);
      camera.position.copy(pose.position);
      camera.lookAt(pose.target);
    } else {
      simTime += dt;
      controls?.update(dt);
    }

    schedule.advance(simTime, pickHeroNeuron);
    if (schedule.dirty) {
      neurons.textures.spikes.needsUpdate = true;
      schedule.dirty = false;
    }
    u.time.value = simTime;
    debug.hero.distanceUm = camera.position.distanceTo(hero.anchor.soma);
    const ndc = hero.anchor.soma.clone().project(camera);
    debug.hero.screen = [((ndc.x + 1) / 2) * canvas.clientWidth, ((1 - ndc.y) / 2) * canvas.clientHeight];
    debug.simTime = simTime;
    debug.spikes = schedule.count;

    post.render();
    debug.frame++;
  };

  progress(1, "Ready");
  await renderer.setAnimationLoop(loop);

  return {
    isWebGPU,
    dispose() {
      disposed = true;
      abort.abort();
      void renderer.setAnimationLoop(null);
      observer.disconnect();
      controls?.dispose();
      post.dispose();
      neurons.dispose();
      layers.dispose();
      renderer.dispose();
      if (window.__cmm === debug) delete window.__cmm;
    },
  };
}

