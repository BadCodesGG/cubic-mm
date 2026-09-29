/**
 * Wires the experience together: renderer, dataset, scene, camera, post, the spiking
 * simulation, and the animation loop. `startApp` resolves once the first frame is ready to draw.
 *
 * `?shot=hero` runs a fixed-timestep clock and the scripted hero camera, stimulates the hero
 * neuron at t = 0.3 s, and holds the simulation at a fixed time so screenshots are reproducible.
 * `?parity=1` is the same clock run for 5 simulated seconds, for `scripts/parity.mjs`.
 * `?sim=cpu` runs the simulation in the CPU worker even on WebGPU.
 * `?gpuTiming=1` reports the GPU time of each frame's render passes as `__cmm.gpuMs` (WebGPU only).
 * `?quality=hi|lite` and `?pieces=1|2|3` pin the render level (see quality.ts) and turn adapting off;
 * `?budgetMs=N` sets the frame budget adapting steps down against (default 20 ms).
 */

import { PerspectiveCamera, Scene, Vector3 } from "three/webgpu";
import { NO_PARENT, PATH_DIST_UNIT_UM, dequantise } from "./format";
import { loadDataset, type Dataset } from "./data";
import type { Quality } from "./prefs";
import { FrameBudget, maxPixelRatio, resolveQuality, stepDown, type Pieces, type QualityLevel, type Tier } from "./quality";
import { synthDataset } from "./synth";
import { createRenderer } from "./scene/renderer";
import { createNeurons, type Neurons } from "./scene/neurons";
// --- sim (r1/sim): the spiking simulation and its event bus ---
import { EventBus } from "./events";
import { createSimulation, type Simulation } from "./sim";
// --- end sim ---
import { createLayers } from "./scene/layers";
import { createPost } from "./scene/post";
import { createSceneUniforms } from "./scene/uniforms";
// r1-audio: imports
import { createAudio, type AudioEngine } from "./audio/engine";
import { publishAudio, unpublishAudio } from "./audio/handle";
import { FlyControls, heroPath, WORLD_UP, type CameraPose, type HeroAnchor } from "./camera/fly";
// --- r1/nav: navigation imports (picker, ride camera, HUD frame) ---
import { Picker } from "./picker";
import { AHEAD_UM, RideCamera } from "./camera/ride";
import { createHudFrame, updateHudFrame, type HudFrame } from "./camera/view";
import type { Stick } from "./camera/touch";
// --- end r1/nav ---

export interface AppOptions {
  /** "hero" selects the scripted, reproducible camera. */
  shot?: string | null;
  /** Use the synthetic dataset instead of /data. */
  synth?: boolean;
  forceWebGL?: boolean;
  /** Simulation time the hero shot holds at, seconds. */
  holdAt?: number;
  onProgress?: (fraction: number, label: string) => void;
  /** The saved `cmm-quality` preference. */
  quality?: Quality;
  // --- r1/nav ---
  /** "hero" selects the hero neuron at start, so screenshots can show the selection panel. */
  select?: string | null;
  // --- end r1/nav ---
}

export interface App {
  isWebGPU: boolean;
  dispose(): void;
  // --- r1/nav: what the HUD reads and drives ---
  bus: EventBus;
  data: Dataset;
  /** Mutable per-frame numbers for the HUD (scale bar, compass, selection ring, ride state). */
  hud: HudFrame;
  /** Currently selected neuron, -1 for none. */
  selection(): number;
  /** Stop waiting for, or riding, a spike and give the camera back. */
  cancelRide(): void;
  /** Touch pad input: left stick moves, right stick looks. */
  setSticks(move: Stick, look: Stick): void;
  // --- end r1/nav ---
}

export interface CmmDebug {
  isWebGPU: boolean;
  frame: number;
  spikes: number;
  instances: number;
  /** Mean wall-clock ms between the last 60 frames. */
  frameMs: number;
  /** `?gpuTiming=1` only: mean GPU ms of the render passes over the last 60 resolved frames. */
  gpuMs?: number;
  simTime: number;
  /** True once a shot's clock has reached its hold time. */
  settled: boolean;
  dataset: "real" | "synth";
  /** The render level now drawn; `adapted` once the frame-time check has stepped it down. */
  quality: QualityLevel & { instances: number; adapted: boolean };
  hero: { neuron: number; distanceUm: number; screen: [number, number] };
  // --- sim (r1/sim) ---
  sim?: {
    mode: "gpu" | "cpu";
    syntheticSynapses: boolean;
    synapses: number;
    spikes: number;
    /** Spikes caused by a stimulus (the scripted hero, or the visitor). */
    stimulated: number;
    rateHz: number;
    spikeCounts(): Uint32Array;
    /** Post neurons reached by the stimulated neuron's pulses so far. */
    firstHop(): number[];
    /** Resolves once every queued GPU readback or worker reply has been reported. */
    settle(): Promise<void>;
    /** `?parity=1` only: per-frame cost of the simulation at a given size (see sim/bench.ts). */
    bench?(neurons: number, synapses: number): Promise<import("./sim/bench").BenchResult>;
  };
  // --- end sim ---
  // --- r1/nav ---
  selected: number;
  ride: string;
  // --- end r1/nav ---
}

declare global {
  interface Window {
    __cmm?: CmmDebug;
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

export async function startApp(canvas: HTMLCanvasElement, opts: AppOptions = {}): Promise<App> {
  const progress = opts.onProgress ?? (() => {});
  const shot = opts.shot === "hero";
  const abort = new AbortController();

  progress(0.02, "Starting the renderer");
  const params = new URLSearchParams(window.location.search);
  const gpuTiming = params.get("gpuTiming") === "1";
  const { renderer, isWebGPU } = await createRenderer(canvas, { forceWebGL: opts.forceWebGL, trackTimestamp: gpuTiming });
  const parity = params.get("parity") === "1";
  const coarsePointer = window.matchMedia("(pointer: coarse)").matches;
  const pinnedTier = params.get("quality");
  const pinnedPieces = Number(params.get("pieces"));
  const plan = resolveQuality({
    pref: opts.quality ?? "auto",
    isWebGPU,
    coarsePointer,
    viewportWidth: window.innerWidth,
    override: {
      ...(pinnedTier === "hi" || pinnedTier === "lite" ? { tier: pinnedTier } : {}),
      ...(pinnedPieces === 1 || pinnedPieces === 2 || pinnedPieces === 3 ? { pieces: pinnedPieces as Pieces } : {}),
    },
    scripted: shot || parity,
  });
  let level = plan.level;
  const capPixelRatio = () => renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, maxPixelRatio(level, coarsePointer)));
  capPixelRatio();

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
        level.tier,
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

  // --- sim (r1/sim): create the bus and the simulation before the neurons, which read its spikes ---
  const bus = new EventBus();
  const sim: Simulation = createSimulation({
    renderer,
    isWebGPU,
    dataset: data,
    bus,
    seed: 11,
    forceCpu: params.get("sim") === "cpu",
  });
  u.velocity.value = sim.params.conductionMps;
  u.slowMo.value = sim.params.slowMo;
  u.synDelay.value = sim.params.synDelayMs / 1000;
  const stimulate = (neuron: number) => {
    sim.stimulate(neuron);
    u.stimNeuron.value = neuron;
  };
  bus.on("stimulate", ({ neuron }) => stimulate(neuron));
  const buildNeurons = (d: Dataset, pieces: Pieces): Neurons => {
    const built = createNeurons(d, u, pieces, { spikes: sim.spikeTimesSource, synapses: sim.synapses });
    scene.add(built.ribbons, built.somas, built.pulses);
    if (built.synapseGlow) scene.add(built.synapseGlow);
    return built;
  };
  let neurons = buildNeurons(data, level.pieces);
  // Soma positions are the same in every LOD, so this one array serves the picker across rebuilds.
  const somaPos = neurons.somaPos;
  // --- end sim ---
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

  // --- sim (r1/sim) ---
  const scripted = shot || parity;
  const holdAt = parity ? 5 : (opts.holdAt ?? 1.3);
  /** Scripted runs stimulate the hero neuron once, at this simulation time. */
  const HERO_STIMULUS_AT = 0.3;
  // --- end sim ---

  // --- r1/nav: picker, ride camera, HUD frame (the bus is created with the simulation above) ---
  const hud = createHudFrame();
  let selected = -1;
  bus.on("select", (e) => {
    selected = e.neuron;
    u.selectedNeuron.value = e.neuron;
  });
  // The ride is created first so its Esc handler runs before the picker's and can mark the key consumed.
  const ride = new RideCamera(data, bus, camera, controls);
  const picker = new Picker(canvas, camera, somaPos, bus);
  if (controls) controls.canLock = (e) => picker.pickAt(e) < 0;
  const focus = new Vector3();
  const selectedSoma = new Vector3();
  const rideDir = new Vector3();
  const heroNdc = new Vector3();
  // --- end r1/nav ---
  const heroNeuron = hero.neuron;
  // --- r1-audio: synthesised spatial sound, subscribed to the bus ---
  const audio: AudioEngine = createAudio({ bus, dataset: data, getCamera: () => camera });
  publishAudio(audio);
  // --- end r1-audio ---

  const debug: CmmDebug = {
    isWebGPU,
    frame: 0,
    spikes: 0,
    instances: neurons.instanceCount,
    frameMs: 0,
    simTime: 0,
    settled: false,
    dataset: source,
    quality: { ...level, instances: neurons.instanceCount, adapted: false },
    hero: { neuron: hero.neuron, distanceUm: 0, screen: [0, 0] },
    // --- sim (r1/sim) ---
    sim: {
      mode: sim.mode,
      syntheticSynapses: sim.syntheticSynapses,
      synapses: sim.synapses.count,
      spikes: 0,
      stimulated: 0,
      rateHz: 0,
      spikeCounts: () => sim.spikeCounts(),
      firstHop: () => sim.firstHop(),
      settle: () => sim.settle(),
      bench: parity
        ? async (n, s) => (await import("./sim/bench")).benchSimulation(sim.mode === "gpu" ? renderer : null, n, s)
        : undefined,
    },
    // --- end sim ---
    selected: -1,
    ride: "idle",
  };
  window.__cmm = debug;

  // --- r2/quality: step the render level down while the frame time is over budget ---
  // The ribbons are rebuilt in place; the simulation, camera and HUD keep running. A step to the
  // lite LOD fetches it then, quietly, with no loader; the hi dataset stays the one the HUD,
  // picker and ride read, since soma positions and the synapse table are the same in both.
  const datasets: Partial<Record<Tier, Dataset>> = { [level.tier]: data };
  // `?budgetMs=` lowers the frame budget, so the step-down can be exercised on a fast machine.
  const budgetMs = Number(params.get("budgetMs"));
  let budget = plan.adaptive ? new FrameBudget(budgetMs > 0 ? { budgetMs } : {}) : null;
  let rebuilding = false;
  const rebuild = async (next: QualityLevel) => {
    rebuilding = true;
    try {
      let d = source === "synth" ? data : datasets[next.tier];
      if (!d) {
        d = await loadDataset("/data", next.tier, undefined, abort.signal);
        datasets[next.tier] = d;
      }
      if (disposed) return;
      const old = neurons;
      scene.remove(old.ribbons, old.somas, old.pulses);
      if (old.synapseGlow) scene.remove(old.synapseGlow);
      old.dispose();
      neurons = buildNeurons(d, next.pieces);
      level = next;
      capPixelRatio();
      resize();
      debug.instances = neurons.instanceCount;
      debug.quality = { ...level, instances: neurons.instanceCount, adapted: true };
      budget?.reset();
    } catch (err) {
      if (!disposed) console.warn("Could not step the render quality down:", err);
    } finally {
      rebuilding = false;
    }
  };
  // --- end r2/quality ---

  const resize = () => {
    const w = canvas.clientWidth || window.innerWidth;
    const h = canvas.clientHeight || window.innerHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    const bufferHeight = h * renderer.getPixelRatio();
    u.pixelScale.value = camera.projectionMatrix.elements[5] * 0.5 * bufferHeight;
    u.minHalfWidthPx.value = 0.8 * renderer.getPixelRatio();
    u.glowCapPx.value = 12 * renderer.getPixelRatio();
  };
  resize();
  const observer = new ResizeObserver(resize);
  observer.observe(canvas);

  let simTime = 0;
  let last = -1;
  const deltas: number[] = [];
  let disposed = false;
  let timingPending = false;
  const gpuSamples: number[] = [];
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
      // --- r2/quality ---
      if (budget && !rebuilding && budget.push(now - last) === "slow") {
        const next = stepDown(level);
        if (next) void rebuild(next);
        else budget = null;
      }
      // --- end r2/quality ---
    }
    last = now;

    const prevSimTime = simTime;
    if (scripted) {
      simTime = Math.min(debug.frame / 60, holdAt);
      debug.settled = simTime >= holdAt;
      heroPath(simTime, hero.anchor, pose);
      camera.position.copy(pose.position);
      camera.lookAt(pose.target);
    } else {
      simTime += dt;
      // --- r1/nav: the ride owns the camera while it runs, FlyControls otherwise ---
      if (!ride.update(simTime, dt)) controls?.update(dt);
      // --- end r1/nav ---
    }

    // --- sim (r1/sim) ---
    if (scripted && prevSimTime < HERO_STIMULUS_AT && simTime >= HERO_STIMULUS_AT) stimulate(hero.neuron);
    sim.step(simTime, simTime - prevSimTime);
    neurons.update();
    debug.sim!.spikes = sim.stats.spikes;
    debug.sim!.rateHz = sim.stats.rateHz;
    debug.sim!.stimulated = sim.stats.stimulated;
    // --- end sim ---
    audio.update(dt); // r1-audio
    u.time.value = simTime;
    debug.hero.distanceUm = camera.position.distanceTo(hero.anchor.soma);
    const ndc = heroNdc.copy(hero.anchor.soma).project(camera);
    debug.hero.screen = [((ndc.x + 1) / 2) * canvas.clientWidth, ((1 - ndc.y) / 2) * canvas.clientHeight];
    debug.simTime = simTime;
    debug.spikes = sim.stats.spikes;

    // --- r1/nav: HUD numbers for this frame ---
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    const focusNeuron = selected >= 0 ? selected : heroNeuron;
    focus.fromArray(somaPos, focusNeuron * 3);
    // On a ride the camera is inside the axon: measure the scale at the point being looked at, not at the far soma.
    if (ride.active) focus.copy(camera.position).addScaledVector(camera.getWorldDirection(rideDir), AHEAD_UM);
    if (selected >= 0) selectedSoma.fromArray(somaPos, selected * 3);
    updateHudFrame(hud, camera, w, h, focus, selected >= 0 ? selectedSoma : null);
    hud.ride = ride.state;
    hud.rideNeuron = ride.neuron;
    hud.locked = document.pointerLockElement === canvas;
    debug.selected = selected;
    debug.ride = ride.state;
    // --- end r1/nav ---

    post.render();
    debug.frame++;
    if (gpuTiming && isWebGPU && !timingPending) {
      timingPending = true;
      void renderer.resolveTimestampsAsync("render").then((ms) => {
        timingPending = false;
        if (typeof ms !== "number" || ms <= 0) return;
        gpuSamples.push(ms);
        if (gpuSamples.length > 60) gpuSamples.shift();
        debug.gpuMs = gpuSamples.reduce((a, b) => a + b, 0) / gpuSamples.length;
      });
    }
  };

  progress(1, "Ready");
  await renderer.setAnimationLoop(loop);
  // --- r1/nav ---
  if (opts.select === "hero") bus.emit("select", { neuron: hero.neuron });
  // --- end r1/nav ---

  return {
    isWebGPU,
    // --- r1/nav ---
    bus,
    data,
    hud,
    selection: () => selected,
    cancelRide: () => ride.cancel(),
    setSticks: (move, look) => controls?.setSticks(move, look),
    // --- end r1/nav ---
    dispose() {
      // --- r1/nav ---
      ride.dispose();
      picker.dispose();
      // --- end r1/nav ---
      disposed = true;
      abort.abort();
      void renderer.setAnimationLoop(null);
      observer.disconnect();
      controls?.dispose();
      unpublishAudio(audio); // r1-audio
      audio.dispose(); // r1-audio
      post.dispose();
      neurons.dispose();
      // --- sim (r1/sim) ---
      sim.dispose();
      bus.clear();
      // --- end sim ---
      layers.dispose();
      renderer.dispose();
      if (window.__cmm === debug) delete window.__cmm;
    },
  };
}

