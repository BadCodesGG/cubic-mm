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
 * `?tour=1` plays the intro tour even when it has been seen; with `&t=N` it runs on a fixed clock
 * and holds at N tour seconds, so a shot of any moment of the tour is reproducible.
 * `?story=<id>` plays a guided story (stories.ts); with `&t=N` it holds at N story seconds the same way.
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
// --- r3/tour: intro tour, home and jump flights, time control, screenshots ---
import { Flight, poseAround, type Pose } from "./camera/goto";
import { markTourSeen, readTourSeen } from "./prefs";
import { Tour, introFade, shouldRunTour } from "./tour";
import { captureFrame, downloadBlob, shotFilename } from "./screenshot";
// --- end r3/tour
// --- r3/links: shareable links ---
import { LinkSync, decodeView, lookAngles, lookDirection, type ViewState } from "./share";
// --- end r3/links ---
// --- r3/cascade ---
import { CascadeTracker } from "./cascade";
import { createCascadeLines } from "./scene/cascade";
// --- end r3/cascade ---
// --- r4/graph: the connectome as a graph (hover and pinned wiring) ---
import { PREVIEW_S, WiringState, buildPartnerIndex, type PartnerIndex } from "./partners";
import { createPartnerGraph } from "./scene/partners";
// --- end r4/graph ---
// --- r4/stories ---
import { StoryPlayer, buildStories } from "./stories";
import { createInhibitionMarks } from "./scene/neurons";
// --- end r4/stories

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
  // --- r3/tour ---
  /** `?tour=1`: play the intro tour even if seen (never in a scripted shot or parity run). */
  tour?: boolean;
  // --- end r3/tour
}

// --- r3/tour ---
export interface TourHandle {
  running(): boolean;
  /** Seconds since the tour started. */
  time(): number;
  heroNeuron: number;
  skip(): void;
  /** Plays the tour again from the start ("Replay the intro"). */
  replay(): void;
}
// --- end r3/tour

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
  // --- r3/tour ---
  tour: TourHandle;
  /** Current simulation speed: 1 normal, 0.1 slow motion, 0 paused. */
  timeScale(): number;
  /** Called after each screenshot with its filename, or null if it could not be taken. */
  onScreenshot(listener: (filename: string | null) => void): () => void;
  // --- end r3/tour
  // --- r3/links ---
  /** The current view as a link: writes the URL hash now, then returns `location.href`. */
  shareUrl(): string;
  /** Camera position in µm. The array is reused; read it, do not keep it. */
  cameraPosition(): readonly [number, number, number];
  // --- end r3/links ---
  // --- r3/cascade ---
  /** What the last stimulus set off: hop counts, edges and a timeline. */
  cascade: CascadeTracker;
  // --- end r3/cascade ---
  // --- r4/graph ---
  /** Who each cell synapses onto and receives from; null when the dataset has no synapse table. */
  partners: PartnerIndex | null;
  /** Which cell's graph is pinned (`pinned`), or -1. */
  wiring: WiringState;
  // --- end r4/graph ---
  // --- r4/stories ---
  stories: StoriesHandle;
  // --- end r4/stories
}

// --- r4/stories ---
export interface StoriesHandle {
  /** The stories this dataset can tell, in menu order. */
  list: readonly { id: string; title: string; blurb: string }[];
  running(): boolean;
  /** The story playing, or null. */
  id(): string | null;
  /** The caption on screen now, or null. */
  caption(): string | null;
  start(id: string): void;
  skip(): void;
}
// --- end r4/stories

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
  // --- r3/tour ---
  /** Camera position (µm) and heading, `yaw = atan2(dir.x, dir.z)`, for the minimap. */
  camera?: { x: number; y: number; z: number; yaw: number };
  timeScale?: number;
  tour?: { running: boolean; time: number };
  // --- end r3/tour
  // --- r3/links ---
  /** The camera and selection as a link would carry them (`share.ts`). Absent in a scripted shot. */
  view?: () => ViewState;
  // --- end r3/links ---
  // --- r3/cascade ---
  /** The cascade tracker's summary, for checks. */
  cascade?: () => import("./cascade").CascadeSummary;
  /** Selects and stimulates a neuron, as a click and Space would. */
  stimulate?: (neuron: number) => void;
  // --- end r3/cascade ---
  // --- r4/graph: for scripts/shot.mjs --hover and --partners, and for checks ---
  wiring?: {
    /** ms the partner index took to build. */
    buildMs: number;
    /** Most partners (outputs plus inputs) of any cell. */
    maxPartners: number;
    /** Lines drawn this frame. */
    lines(): number;
    /** Emits `hover`, as the pointer over a soma does. */
    hover(neuron: number): void;
    /** Selects the cell and pins (or unpins) its graph. */
    pin(neuron: number, show?: boolean): void;
    state(): { hover: number; pinned: number };
  };
  // --- end r4/graph ---
  // --- r4/stories ---
  story?: { id: string | null; time: number; caption: string | null };
  // --- end r4/stories
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
  // --- r3/cascade: reads stimulate, spike and arrive; "now" is the simulation clock (simTime, declared below) ---
  const cascade = new CascadeTracker(bus, { clock: () => simTime, inhibitory: data.neurons.inhibitory });
  const cascadeLines = createCascadeLines(data, cascade, u);
  scene.add(cascadeLines.object);
  // --- end r3/cascade ---
  // --- r4/graph: the index is built once, here; hover, pin and preview are read from the bus ---
  const indexStart = performance.now();
  const partnerIndex = data.synapses ? buildPartnerIndex(data.synapses, data.neurons.inhibitory) : null;
  const partnerBuildMs = performance.now() - indexStart;
  const wiring = new WiringState(bus, data.neurons.count);
  const partnerGraph = partnerIndex ? createPartnerGraph(data, partnerIndex, wiring, u) : null;
  if (partnerGraph) scene.add(partnerGraph.group);
  const onWiringKey = (e: KeyboardEvent) => {
    if (!scripted) wiring.onKey(e, bus);
  };
  window.addEventListener("keydown", onWiringKey);
  // --- end r4/graph ---
  // --- r4/stories: where an inhibitory story cell's pulses just landed; kept across quality rebuilds ---
  const inhibitionMarks = createInhibitionMarks(data.neurons.count);
  bus.on("arrive", ({ pre, post, time }) => {
    if (pre === u.inhibitNeuron.value) inhibitionMarks.mark(post, time);
  });
  // --- end r4/stories
  const buildNeurons = (d: Dataset, pieces: Pieces): Neurons => {
    const built = createNeurons(d, u, pieces, { spikes: sim.spikeTimesSource, synapses: sim.synapses, inhibition: inhibitionMarks });
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

  // --- r3/links: the URL hash carries the camera and the selection (share.ts) ---
  const linkDir = new Vector3();
  const currentView = (): ViewState => {
    camera.getWorldDirection(linkDir);
    const { yaw, pitch } = lookAngles(linkDir.x, linkDir.y, linkDir.z);
    return { x: camera.position.x, y: camera.position.y, z: camera.position.z, yaw, pitch, neuron: selected };
  };
  // A scripted shot keeps its own camera and its own URL.
  const link = scripted || !controls
    ? null
    : new LinkSync({
        getView: currentView,
        count: data.neurons.count,
        // replaceState, never pushState: the address bar follows the view without filling the back button.
        write: (hash) => history.replaceState(history.state, "", `${location.pathname}${location.search}#${hash}`),
      });
  /** Put the camera where a link says, at once and with no flight. */
  const placeCamera = (v: ViewState) => {
    ride.cancel();
    camera.position.set(v.x, v.y, v.z);
    const [dx, dy, dz] = lookDirection(v.yaw, v.pitch);
    controls?.lookAt(linkDir.set(v.x + dx, v.y + dy, v.z + dz));
  };
  const restored = link ? decodeView(location.hash, data.neurons.count) : null;
  if (restored) placeCamera(restored);
  bus.on("select", () => link?.flush());
  // The tour's cameras are not links. When it ends, the sync resumes and re-bases on the resting view
  // (in the loop, once a skip's flight to rest has landed), so a visitor who never moved keeps a clean URL.
  let linkResume = false;
  bus.on("tour", (e) => {
    linkResume = !e.running;
    if (e.running) link?.pause(true);
  });
  // A link pasted into this tab's address bar changes only the hash, so the page does not reload.
  const onHashChange = () => {
    const v = decodeView(location.hash, data.neurons.count);
    if (!v) return;
    placeCamera(v);
    if (v.neuron !== selected) bus.emit("select", { neuron: v.neuron });
  };
  if (link) window.addEventListener("hashchange", onHashChange);
  const cameraPos: [number, number, number] = [0, 0, 0];
  // --- end r3/links ---

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
    view: scripted ? undefined : currentView, // r3/links
    // --- r3/cascade: for scripts/shot.mjs --stimulate, which picks a well-connected cell ---
    cascade: () => cascade.summary(),
    stimulate: (neuron) => {
      bus.emit("select", { neuron });
      bus.emit("stimulate", { neuron });
    },
    // --- end r3/cascade ---
    // --- r4/graph ---
    wiring: {
      buildMs: partnerBuildMs,
      maxPartners: partnerIndex?.maxPartners ?? 0,
      lines: () => partnerGraph?.lineCount ?? 0,
      hover: (neuron) => bus.emit("hover", { neuron }),
      pin: (neuron, show = true) => {
        bus.emit("select", { neuron });
        bus.emit("partners", { neuron, show });
      },
      state: () => ({ hover: wiring.shown().hover, pinned: wiring.pinned }),
    },
    // --- end r4/graph ---
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

  // --- r3/tour: one shared flight for home, jump and the tour; the time scale; screenshots ---
  const flight = new Flight(camera, controls);
  const restPose: Pose = { position: new Vector3(), target: new Vector3() };
  heroPath(0, hero.anchor, restPose);
  let timeScale = 1;
  bus.on("timeScale", ({ scale }) => {
    if (Number.isFinite(scale) && scale >= 0) timeScale = scale;
  });
  bus.on("home", () => {
    if (scripted) return;
    ride.cancel();
    flight.start(restPose);
  });
  const jumpPose: Pose = { position: new Vector3(), target: new Vector3() };
  const jumpSoma = new Vector3();
  const jumpAway = new Vector3();
  bus.on("jump", ({ neuron }) => {
    if (scripted || !(neuron >= 0 && neuron < data.neurons.count)) return;
    ride.cancel();
    jumpSoma.fromArray(somaPos, neuron * 3);
    // Arrive from the side the camera is already on, level with the soma; poseAround adds the lift.
    jumpAway.copy(camera.position).sub(jumpSoma).setY(0);
    if (jumpAway.lengthSq() < 1e-6) jumpAway.copy(hero.anchor.axis).negate().setY(0);
    jumpAway.normalize();
    flight.start(poseAround(jumpSoma, jumpAway, 120, jumpPose), undefined, () => {
      bus.emit("select", { neuron });
      wiring.preview(neuron, PREVIEW_S); // r4/graph: the search reward, a graph for a few seconds
    });
  });
  // A ride takes the camera from a flight in progress.
  bus.on("ride", () => flight.cancel());

  const tour = new Tour({ flight, camera, bus, heroNeuron: hero.neuron, anchor: hero.anchor, onEnd: () => markTourSeen() });
  const startTour = () => {
    if (scripted) return;
    ride.cancel();
    // Start first: it pauses the link sync, so the deselect below does not write a hash.
    tour.start();
    bus.emit("select", { neuron: -1 });
    bus.emit("timeScale", { scale: 1 });
  };
  /** `?tour=1&t=N`: the tour on a fixed 60 Hz clock, held at N seconds. */
  const tourHoldAt = opts.tour && opts.holdAt !== undefined && !scripted ? opts.holdAt : null;
  // Input that skips the tour does nothing else; buttons (Skip itself, the sound toggle) keep working.
  const MODIFIERS = new Set(["ShiftLeft", "ShiftRight", "ControlLeft", "ControlRight", "AltLeft", "AltRight", "MetaLeft", "MetaRight"]);
  let swallowClickUntil = 0;
  const onTourInput = (e: Event) => {
    if (e.type === "click") {
      if (performance.now() < swallowClickUntil) e.stopPropagation();
      return;
    }
    if (!tour.running || tourHoldAt !== null) return;
    if (e.target instanceof Element && e.target.closest("button, a, [role='button']")) return;
    if (e instanceof KeyboardEvent && (MODIFIERS.has(e.code) || e.ctrlKey || e.metaKey || e.altKey)) return;
    e.stopPropagation();
    if (e.type === "keydown") e.preventDefault();
    if (e.type === "pointerdown" || e.type === "touchstart") swallowClickUntil = performance.now() + 1000;
    tour.skip();
  };
  const TOUR_INPUT = ["keydown", "pointerdown", "click", "wheel", "touchstart"] as const;
  for (const type of TOUR_INPUT) {
    window.addEventListener(type, onTourInput, { capture: true, passive: type === "wheel" || type === "touchstart" });
  }

  const onAppKey = (e: KeyboardEvent) => {
    if (scripted || e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.target instanceof Element && e.target.closest("input, textarea, select, [contenteditable]")) return;
    if (e.code === "KeyH") bus.emit("home", {});
    else if (e.code === "KeyP") bus.emit("screenshot", {});
    else if (e.code === "Comma") bus.emit("timeScale", { scale: 0.1 });
    else if (e.code === "Period") bus.emit("timeScale", { scale: 1 });
  };
  window.addEventListener("keydown", onAppKey);

  // --- r4/stories: guided stories, played like the tour through the same flight ---
  const storyDefs = buildStories(
    data,
    { reached: () => sim.firstHop().length, cascade: () => cascade.summary() },
    { conductionMps: sim.params.conductionMps, slowMo: sim.params.slowMo, synDelayS: sim.params.synDelayMs / 1000 },
  );
  const story = new StoryPlayer({
    bus,
    fly: (p, seconds) => {
      ride.cancel();
      flight.start(p, seconds);
    },
    place: (p) => {
      flight.cancel();
      ride.cancel();
      camera.position.copy(p.position);
      camera.up.copy(WORLD_UP);
      camera.lookAt(p.target);
      // The controls run between flights; they must look where the orbit looks.
      controls?.lookAt(p.target);
    },
    rest: () => restPose,
    // A story that plays out hands the visitor its cell, selected, where it ended.
    onEnd: (id, reason) => {
      const def = storyDefs.find((d) => d.id === id);
      if (def && reason === "done") bus.emit("select", { neuron: def.neuron });
    },
  });
  const storyParam = params.get("story");
  /** `?story=<id>&t=N`: the story on a fixed 60 Hz clock, held N real seconds in (waits included). */
  const storyHoldAt = storyParam && opts.holdAt !== undefined && !scripted ? opts.holdAt : null;
  const startStory = (id: string) => {
    const def = storyDefs.find((d) => d.id === id);
    if (scripted || !def) return;
    if (tour.running) tour.skip();
    ride.cancel();
    // Start first: it pauses the link sync, so the deselect below does not write a hash.
    story.start(def);
    bus.emit("select", { neuron: -1 });
    bus.emit("timeScale", { scale: 1 });
  };
  bus.on("story", (e) => {
    // The story's cell is drawn through the haze, and an inhibitory one's pulses land violet.
    const def = e.id === null ? null : storyDefs.find((d) => d.id === e.id);
    u.focusNeuron.value = def ? def.neuron : -1;
    u.inhibitNeuron.value = def && data.neurons.inhibitory[def.neuron] && def.id === "basket" ? def.neuron : -1;
    inhibitionMarks.clear();
    // Story cameras are not links either; the sync resumes once the last flight lands.
    linkResume = e.id === null && !tour.running;
    if (e.id !== null) link?.pause(true);
  });
  // Replaying the intro ends a story.
  bus.on("tour", (e) => {
    if (e.running && story.running) story.cancel();
  });
  // Any input skips a story, as it does the tour; buttons keep working.
  const onStoryInput = (e: Event) => {
    if (e.type === "click" || !story.running || storyHoldAt !== null) return;
    if (e.target instanceof Element && e.target.closest("button, a, [role='button']")) return;
    if (e instanceof KeyboardEvent && (MODIFIERS.has(e.code) || e.ctrlKey || e.metaKey || e.altKey)) return;
    e.stopPropagation();
    if (e.type === "keydown") e.preventDefault();
    if (e.type === "pointerdown" || e.type === "touchstart") swallowClickUntil = performance.now() + 1000;
    story.skip();
  };
  for (const type of TOUR_INPUT) {
    window.addEventListener(type, onStoryInput, { capture: true, passive: type === "wheel" || type === "touchstart" });
  }
  const debugStory: NonNullable<CmmDebug["story"]> = { id: null, time: 0, caption: null };
  debug.story = debugStory;
  // --- end r4/stories

  const shotListeners = new Set<(filename: string | null) => void>();
  let shotPending = false;
  bus.on("screenshot", () => {
    shotPending = true;
  });
  /** Runs inside the animation loop, so the high-resolution render and `toBlob` share one task. */
  const takeScreenshot = () => {
    const filename = shotFilename(selected >= 0 ? data.neurons.rootId[selected].toString() : null);
    const report = (blob: Blob | null) => {
      if (blob) downloadBlob(blob, filename);
      for (const listener of shotListeners) listener(blob ? filename : null);
    };
    try {
      void captureFrame({
        canvas,
        getPixelRatio: () => renderer.getPixelRatio(),
        setPixelRatio: (ratio) => renderer.setPixelRatio(ratio),
        resize,
        render: () => post.render(),
      }).then(report, () => report(null));
    } catch (err) {
      console.warn("Could not take the screenshot:", err);
      report(null);
    }
    // The capture frame is slow by design: keep it out of the frame budget and the next frame's dt.
    last = -1;
    budget?.reset();
  };
  const camDir = new Vector3();
  const debugCamera = { x: 0, y: 0, z: 0, yaw: 0 };
  debug.camera = debugCamera;
  const debugTour = { running: false, time: 0 };
  debug.tour = debugTour;
  // --- end r3/tour

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
      // --- r3/tour: the time scale slows the simulation only; the tour and flights keep real time ---
      let step = tourHoldAt === null ? dt : tour.time < tourHoldAt - 1e-9 ? 1 / 60 : 0;
      if (tourHoldAt !== null) debug.settled = step === 0;
      // --- r4/stories: the story's timeline runs before the flights it starts are advanced ---
      if (storyHoldAt !== null) {
        step = story.elapsed < storyHoldAt - 1e-9 ? 1 / 60 : 0;
        debug.settled = step === 0;
      }
      story.update(step);
      debugStory.id = story.id;
      debugStory.time = story.time;
      debugStory.caption = story.caption();
      // --- end r4/stories
      simTime += step * timeScale;
      tour.update(step);
      // The tour opens dim and fades up; after a skip the light returns over a second instead of jumping.
      u.introFade.value = tour.running ? introFade(tour.time) : Math.min(1, u.introFade.value + step);
      // --- end r3/tour
      // --- r1/nav: the ride owns the camera while it runs, then a flight (r3/tour), FlyControls otherwise ---
      if (!ride.update(simTime, step) && !flight.update(step)) controls?.update(step);
      // --- end r1/nav ---
    }

    // --- sim (r1/sim) ---
    // --- r3/cascade: through the bus, so the cascade tracker sees the scripted stimulus too ---
    if (scripted && prevSimTime < HERO_STIMULUS_AT && simTime >= HERO_STIMULUS_AT) bus.emit("stimulate", { neuron: hero.neuron });
    // --- end r3/cascade ---
    sim.step(simTime, simTime - prevSimTime);
    neurons.update();
    debug.sim!.spikes = sim.stats.spikes;
    debug.sim!.rateHz = sim.stats.rateHz;
    debug.sim!.stimulated = sim.stats.stimulated;
    // --- end sim ---
    audio.update(dt); // r1-audio
    u.time.value = simTime;
    cascadeLines.update(simTime); // r3/cascade
    // --- r4/graph: real seconds, so the fades and the preview timer ignore the time scale and a paused sim ---
    wiring.tick(dt);
    partnerGraph?.update(dt);
    // --- end r4/graph ---
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
    // --- r3/tour ---
    camera.getWorldDirection(camDir);
    debugCamera.x = camera.position.x;
    debugCamera.y = camera.position.y;
    debugCamera.z = camera.position.z;
    debugCamera.yaw = Math.atan2(camDir.x, camDir.z);
    debug.timeScale = scripted ? 1 : timeScale;
    debugTour.running = tour.running;
    debugTour.time = tour.time;
    if (shotPending) {
      shotPending = false;
      takeScreenshot();
    }
    // --- end r3/tour
    // --- r3/links ---
    if (linkResume && !flight.active) {
      linkResume = false;
      link?.rebase();
      link?.pause(false);
    }
    link?.update(now);
    // --- end r3/links ---

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
  // --- r3/tour: the tour places the camera before the first frame is drawn ---
  if (shouldRunTour({ seen: readTourSeen(), shot, parity, hash: window.location.hash, forced: opts.tour === true })) startTour();
  // --- end r3/tour
  if (storyParam) startStory(storyParam); // r4/stories
  await renderer.setAnimationLoop(loop);
  // --- r1/nav ---
  if (opts.select === "hero") bus.emit("select", { neuron: hero.neuron });
  // --- end r1/nav ---
  // --- r3/links ---
  if (restored && restored.neuron >= 0) bus.emit("select", { neuron: restored.neuron });
  // --- end r3/links ---

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
    // --- r3/tour ---
    tour: {
      running: () => tour.running,
      time: () => tour.time,
      heroNeuron: hero.neuron,
      skip: () => tour.skip(),
      replay: startTour,
    },
    timeScale: () => (scripted ? 1 : timeScale),
    onScreenshot(listener) {
      shotListeners.add(listener);
      return () => shotListeners.delete(listener);
    },
    // --- end r3/tour
    // --- r3/links ---
    shareUrl: () => {
      link?.flush();
      return location.href;
    },
    cameraPosition: () => {
      cameraPos[0] = camera.position.x;
      cameraPos[1] = camera.position.y;
      cameraPos[2] = camera.position.z;
      return cameraPos;
    },
    // --- end r3/links ---
    cascade, // r3/cascade
    partners: partnerIndex, // r4/graph
    wiring, // r4/graph
    // --- r4/stories ---
    stories: {
      list: storyDefs.map(({ id, title, blurb }) => ({ id, title, blurb })),
      running: () => story.running,
      id: () => story.id,
      caption: () => story.caption(),
      start: startStory,
      skip: () => story.skip(),
    },
    // --- end r4/stories
    dispose() {
      // --- r1/nav ---
      ride.dispose();
      picker.dispose();
      // --- end r1/nav ---
      window.removeEventListener("hashchange", onHashChange); // r3/links
      disposed = true;
      // --- r3/tour ---
      for (const type of TOUR_INPUT) window.removeEventListener(type, onTourInput, { capture: true });
      window.removeEventListener("keydown", onAppKey);
      // --- r4/stories ---
      for (const type of TOUR_INPUT) window.removeEventListener(type, onStoryInput, { capture: true });
      inhibitionMarks.dispose();
      // --- end r4/stories
      flight.cancel();
      shotListeners.clear();
      // --- end r3/tour
      abort.abort();
      void renderer.setAnimationLoop(null);
      observer.disconnect();
      controls?.dispose();
      unpublishAudio(audio); // r1-audio
      audio.dispose(); // r1-audio
      post.dispose();
      neurons.dispose();
      // --- r3/cascade ---
      cascade.dispose();
      cascadeLines.dispose();
      // --- end r3/cascade ---
      // --- r4/graph ---
      window.removeEventListener("keydown", onWiringKey);
      wiring.dispose();
      partnerGraph?.dispose();
      // --- end r4/graph ---
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

