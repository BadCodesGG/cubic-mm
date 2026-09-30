/**
 * Guided stories: short narrated routes through the volume, each built around a cell chosen from the
 * data at load. Nothing here names a cell by index, and every number a caption quotes is measured.
 *
 * The picks are pure functions over a `Dataset`. Ties go to the lowest neuron index, so a pick is
 * stable across loads.
 */

import { Vector3 } from "three/webgpu";
import { WORLD_UP } from "./camera/fly";
import type { Pose } from "./camera/goto";
import { Compartment, PATH_DIST_UNIT_UM } from "./format";
import type { Dataset } from "./data";
import type { EventBus, EventMap } from "./events";
import { DEFAULT_CONDUCTION_MPS, DEFAULT_SLOW_MO, axonTrack, trackPoint } from "./camera/ride";
import { describeNeuron, formatCount, formatLength } from "./info";
import { layerBoundaries } from "./scene/layers";

const typeOf = (data: Dataset, n: number) => data.manifest.cellTypes[data.neurons.cellType[n]] ?? "";

export interface BasketPick {
  neuron: number;
  /** Every synapse it makes. */
  synapses: number;
  /** Distinct cells those synapses land on. */
  targets: number;
  /** Synapses onto excitatory cells: what it was chosen by. */
  excitatorySynapses: number;
}

/** The basket cell (`BC`) with the most synapses onto excitatory cells. */
export function pickBasketCell(data: Dataset): BasketPick | null {
  const syn = data.synapses;
  if (!syn) return null;
  let best: BasketPick | null = null;
  for (let n = 0; n < data.neurons.count; n++) {
    if (typeOf(data, n) !== "BC") continue;
    const from = syn.preOffsets[n];
    const to = syn.preOffsets[n + 1];
    let excitatory = 0;
    const targets = new Set<number>();
    for (let s = from; s < to; s++) {
      targets.add(syn.post[s]);
      if (!data.neurons.inhibitory[syn.post[s]]) excitatory++;
    }
    if (excitatory > 0 && (!best || excitatory > best.excitatorySynapses)) {
      best = { neuron: n, synapses: to - from, targets: targets.size, excitatorySynapses: excitatory };
    }
  }
  return best;
}

/** Soma to the farthest axon tip along the tree, µm (the stored path distance); 0 without an axon. */
export function longestAxonUm(data: Dataset, neuron: number): number {
  const start = data.neuronNodeStart[neuron];
  const end = start + data.neuronNodeCount[neuron];
  let best = 0;
  for (let i = start; i < end; i++) {
    if (data.nodes.compartment[i] === Compartment.Axon) best = Math.max(best, data.nodes.pathDistQ[i]);
  }
  return best * PATH_DIST_UNIT_UM;
}

/** The neuron whose longest axon branch is the longest in the volume. */
export function pickLongestAxon(data: Dataset): { neuron: number; lengthUm: number } | null {
  let best: { neuron: number; lengthUm: number } | null = null;
  for (let n = 0; n < data.neurons.count; n++) {
    const lengthUm = longestAxonUm(data, n);
    if (lengthUm > 0 && (!best || lengthUm > best.lengthUm)) best = { neuron: n, lengthUm };
  }
  return best;
}

/** The layer 4 pyramidal cell (`4P`) with the most synapses onto layer 2/3 pyramidal cells (`23P`). */
export function pickFeedforward(data: Dataset): { neuron: number; synapses: number; targets: number } | null {
  const syn = data.synapses;
  if (!syn) return null;
  let best: { neuron: number; synapses: number; targets: number } | null = null;
  for (let n = 0; n < data.neurons.count; n++) {
    if (typeOf(data, n) !== "4P") continue;
    let count = 0;
    const targets = new Set<number>();
    for (let s = syn.preOffsets[n]; s < syn.preOffsets[n + 1]; s++) {
      if (typeOf(data, syn.post[s]) !== "23P") continue;
      count++;
      targets.add(syn.post[s]);
    }
    if (count > 0 && (!best || count > best.synapses)) best = { neuron: n, synapses: count, targets: targets.size };
  }
  return best;
}

export interface HubPick {
  neuron: number;
  /** Synapses it receives. */
  inputs: number;
  /** Synapses it makes. */
  outputs: number;
  /** Distinct cells on either side. */
  partners: number;
  /** Distance to the farthest partner's soma, µm. */
  reachUm: number;
}

/** The cell with the most synapses in and out. */
export function pickHub(data: Dataset): HubPick | null {
  const syn = data.synapses;
  if (!syn || syn.count === 0) return null;
  const inputs = new Uint32Array(data.neurons.count);
  for (let s = 0; s < syn.count; s++) inputs[syn.post[s]]++;
  let neuron = 0;
  let bestTotal = -1;
  for (let n = 0; n < data.neurons.count; n++) {
    const total = inputs[n] + syn.preOffsets[n + 1] - syn.preOffsets[n];
    if (total > bestTotal) {
      bestTotal = total;
      neuron = n;
    }
  }
  const partners = new Set<number>();
  for (let s = syn.preOffsets[neuron]; s < syn.preOffsets[neuron + 1]; s++) partners.add(syn.post[s]);
  for (let s = 0; s < syn.count; s++) if (syn.post[s] === neuron) partners.add(syn.pre[s]);
  partners.delete(neuron);
  const soma = data.neurons.somaUm;
  let reachUm = 0;
  for (const p of partners) {
    const d = Math.hypot(soma[p * 3] - soma[neuron * 3], soma[p * 3 + 1] - soma[neuron * 3 + 1], soma[p * 3 + 2] - soma[neuron * 3 + 2]);
    reachUm = Math.max(reachUm, d);
  }
  return { neuron, inputs: inputs[neuron], outputs: bestTotal - inputs[neuron], partners: partners.size, reachUm };
}

// ---------------------------------------------------------------------------------------------
// The player

export interface OrbitAction {
  kind: "orbit";
  centre: Vector3;
  radiusUm: number;
  /** Height above the centre, toward the pia. */
  liftUm: number;
  /** Starting angle in the horizontal (xz) plane, radians. */
  fromAngle: number;
  /** One full turn takes this long. */
  seconds: number;
}

export type StoryAction =
  | { kind: "fly"; pose: Pose; seconds: number }
  /**
   * A function is read when the caption appears, so it can quote what just happened; a `live` one is
   * read every frame it is on screen, so a count can climb.
   */
  | { kind: "caption"; text: string | (() => string); seconds: number; live?: boolean }
  | { kind: "stimulate"; neuron: number }
  /** Asks the ride camera to take the cell's next spike, then fires it. */
  | { kind: "ride"; neuron: number }
  | OrbitAction
  /** Holds the timeline until the event fires or `timeout` real seconds pass. */
  | { kind: "wait"; event: keyof EventMap; timeout: number }
  | { kind: "partners"; neuron: number; show: boolean }
  /** Simulation speed (`timeScale`): 1 normal. Back to 1 when the story ends. */
  | { kind: "speed"; scale: number };

/** One step: `do` runs once when the timeline reaches `at` seconds. */
export interface StoryStep {
  at: number;
  do: StoryAction;
}

export interface StoryDef {
  id: string;
  title: string;
  /** One line for the Stories menu. */
  blurb: string;
  /** The cell the story is about. */
  neuron: number;
  /** The story ends when its timeline reaches this. */
  seconds: number;
  /** Run in order of `at`; steps at the same time run in the order written. */
  steps: StoryStep[];
}

/** The camera pose `k` (0..1) of the way round an orbit, looking at its centre. */
export function orbitPose(orbit: OrbitAction, k: number, out: Pose): Pose {
  const a = orbit.fromAngle + k * Math.PI * 2;
  out.position
    .copy(orbit.centre)
    .add(new Vector3(Math.cos(a) * orbit.radiusUm, 0, Math.sin(a) * orbit.radiusUm))
    .addScaledVector(WORLD_UP, orbit.liftUm);
  out.target.copy(orbit.centre);
  return out;
}

export interface StoryPlayerOptions {
  bus: EventBus;
  /** Start a flight to `pose` over `seconds` (the app's shared `Flight`, taking the camera from a ride). */
  fly(pose: Pose, seconds: number): void;
  /** Put the camera at `pose` now (an orbit, every frame). */
  place(pose: Pose): void;
  /** The resting pose a skip flies back to. */
  rest(): Pose;
  onEnd?(id: string, reason: "done" | "skip"): void;
}

/** A skip flies to the resting pose over this long. */
export const SKIP_FLIGHT_S = 1;

/**
 * Plays a `StoryDef` on real seconds: the app calls `update(dt)` every frame, before the flight.
 * The timeline stands still while a `wait` step is waiting. Emits `story {id}` at start and
 * `story {id: null}` at the end, and takes down any wiring it put up.
 */
export class StoryPlayer {
  private def: StoryDef | null = null;
  private steps: StoryStep[] = [];
  private t = 0;
  private real = 0;
  private next = 0;
  private waiting: { off: () => void; left: number; fired: boolean } | null = null;
  private orbit: { action: OrbitAction; at: number } | null = null;
  private shownCaption: { text: string; read: (() => string) | null; from: number; until: number } | null = null;
  private partnersShown = -1;
  private slowed = false;
  private readonly orbitOut: Pose = { position: new Vector3(), target: new Vector3() };

  constructor(private readonly opts: StoryPlayerOptions) {}

  get running(): boolean {
    return this.def !== null;
  }

  /** The story playing, or null. */
  get id(): string | null {
    return this.def?.id ?? null;
  }

  /** Timeline seconds since the story started (it stands still during a wait). */
  get time(): number {
    return this.t;
  }

  /** Real seconds since the story started, waits included. */
  get elapsed(): number {
    return this.real;
  }

  /** The caption on screen now, or null. */
  caption(): string | null {
    const c = this.shownCaption;
    if (!c || this.t < c.from || this.t >= c.until) return null;
    return c.read ? c.read() : c.text;
  }

  start(def: StoryDef): void {
    if (this.def) this.end("skip");
    this.def = def;
    // Stable, so steps at the same time keep their written order (a ride's caption after the ride).
    this.steps = [...def.steps].sort((a, b) => a.at - b.at);
    this.t = 0;
    this.real = 0;
    this.next = 0;
    this.shownCaption = null;
    this.opts.bus.emit("story", { id: def.id });
    this.runDue();
  }

  update(dt: number): void {
    if (!this.def) return;
    this.real += dt;
    if (this.waiting) {
      this.waiting.left -= dt;
      if (!this.waiting.fired && this.waiting.left > 0) return;
      this.stopWaiting();
    } else {
      this.t += dt;
    }
    this.runDue();
    if (!this.def) return;
    if (this.orbit) {
      const k = Math.min(1, (this.t - this.orbit.at) / this.orbit.action.seconds);
      this.opts.place(orbitPose(this.orbit.action, k, this.orbitOut));
      if (k >= 1) this.orbit = null;
    }
    if (this.t >= this.def.seconds && !this.waiting) this.end("done");
  }

  /** Ends the story now and flies to the resting pose. A no-op when none is playing. */
  skip(): void {
    if (!this.def) return;
    this.opts.fly(this.opts.rest(), SKIP_FLIGHT_S);
    this.end("skip");
  }

  /** Ends the story now and leaves the camera where it is (another sequence is taking it over). */
  cancel(): void {
    if (this.def) this.end("skip");
  }

  private runDue(): void {
    const steps = this.steps;
    while (!this.waiting && this.next < steps.length && steps[this.next].at <= this.t + 1e-9) {
      const step = steps[this.next++];
      this.run(step);
    }
  }

  private run(step: StoryStep): void {
    const { bus } = this.opts;
    const a = step.do;
    switch (a.kind) {
      case "fly":
        this.orbit = null;
        this.opts.fly(a.pose, a.seconds);
        break;
      case "caption":
        this.shownCaption = {
          text: typeof a.text === "function" ? a.text() : a.text,
          read: a.live && typeof a.text === "function" ? a.text : null,
          from: step.at,
          until: step.at + a.seconds,
        };
        break;
      case "stimulate":
        bus.emit("stimulate", { neuron: a.neuron });
        break;
      case "ride":
        bus.emit("ride", { neuron: a.neuron });
        bus.emit("stimulate", { neuron: a.neuron });
        break;
      case "orbit":
        this.orbit = { action: a, at: step.at };
        break;
      case "wait": {
        // The timeline stops at the wait's own time, so the steps after it keep their spacing.
        this.t = step.at;
        const w = { off: () => {}, left: a.timeout, fired: false };
        w.off = bus.on(a.event, () => {
          w.fired = true;
        });
        this.waiting = w;
        break;
      }
      case "partners":
        bus.emit("partners", { neuron: a.neuron, show: a.show });
        this.partnersShown = a.show ? a.neuron : -1;
        break;
      case "speed":
        bus.emit("timeScale", { scale: a.scale });
        this.slowed = a.scale !== 1;
        break;
    }
  }

  private stopWaiting(): void {
    this.waiting?.off();
    this.waiting = null;
  }

  private end(reason: "done" | "skip"): void {
    const def = this.def!;
    this.stopWaiting();
    this.orbit = null;
    this.shownCaption = null;
    if (this.partnersShown >= 0) {
      this.opts.bus.emit("partners", { neuron: this.partnersShown, show: false });
      this.partnersShown = -1;
    }
    if (this.slowed) {
      this.opts.bus.emit("timeScale", { scale: 1 });
      this.slowed = false;
    }
    this.def = null;
    this.opts.bus.emit("story", { id: null });
    this.opts.onEnd?.(def.id, reason);
  }
}

// ---------------------------------------------------------------------------------------------
// The four stories

/** What a caption can quote about what just happened, read when the caption appears. */
export interface StoryLive {
  /** Distinct cells the last stimulated cell's pulses have landed on (`sim.firstHop()`). */
  reached(): number;
  /** The cascade tracker's summary of the last stimulus. */
  cascade(): { hop1Cells: readonly number[]; total: number };
}

export interface StoryConduction {
  conductionMps: number;
  slowMo: number;
  /** Synaptic delay, seconds. */
  synDelayS: number;
}

const DEFAULT_CONDUCTION: StoryConduction = { conductionMps: DEFAULT_CONDUCTION_MPS, slowMo: DEFAULT_SLOW_MO, synDelayS: 0.001 };
/** Half the camera's vertical field of view (55 degrees). */
const HALF_FOV = (55 / 2) * (Math.PI / 180);

const somaOf = (data: Dataset, n: number) => new Vector3().fromArray(data.neurons.somaUm, n * 3);
const pose = (position: Vector3, target: Vector3): Pose => ({ position, target });

/** Horizontal unit vector (y is depth), or +x when `v` is vertical. */
function level(v: Vector3): Vector3 {
  const h = new Vector3(v.x, 0, v.z);
  return h.lengthSq() > 1e-6 ? h.normalize() : new Vector3(1, 0, 0);
}

/** `v` turned by `radians` about the vertical. */
function turn(v: Vector3, radians: number): Vector3 {
  return v.clone().applyAxisAngle(WORLD_UP, radians);
}

/** Per-axis median of some somas: the middle of a group, without its stragglers pulling it. */
function medianSoma(data: Dataset, cells: Iterable<number>): Vector3 {
  const axes: number[][] = [[], [], []];
  for (const n of cells) for (let k = 0; k < 3; k++) axes[k].push(data.neurons.somaUm[n * 3 + k]);
  const mid = (v: number[]) => v.sort((a, b) => a - b)[v.length >> 1];
  return new Vector3(mid(axes[0]), mid(axes[1]), mid(axes[2]));
}

/** Post cells of `pre`'s synapses that pass `keep`, and the farthest of those synapses along the axon, µm. */
function targetsOf(data: Dataset, pre: number, keep: (post: number) => boolean): { cells: Set<number>; farthestUm: number } {
  const syn = data.synapses!;
  const cells = new Set<number>();
  let farthestUm = 0;
  for (let s = syn.preOffsets[pre]; s < syn.preOffsets[pre + 1]; s++) {
    if (!keep(syn.post[s])) continue;
    cells.add(syn.post[s]);
    farthestUm = Math.max(farthestUm, syn.preDistQ[s] * PATH_DIST_UNIT_UM);
  }
  return { cells, farthestUm };
}

/** Seconds from a spike until a pulse has landed `um` along the axon. */
const landsAfter = (um: number, c: StoryConduction) => um / ((c.conductionMps * 1e6) / c.slowMo) + c.synDelayS;

/** "layer 4 pyramidal" reads as "layer 4 pyramidal cell"; "basket cell" stays as it is. */
const cellNoun = (typeName: string) => (typeName.endsWith("cell") ? typeName : `${typeName} cell`);

const byTime = (steps: StoryStep[]) => steps.sort((a, b) => a.at - b.at);

function basketStory(data: Dataset, live: StoryLive, c: StoryConduction): StoryDef | null {
  const pick = pickBasketCell(data);
  if (!pick) return null;
  const soma = somaOf(data, pick.neuron);
  const { cells, farthestUm } = targetsOf(data, pick.neuron, () => true);
  const toward = medianSoma(data, cells);
  // From the far side of the soma, looking through it at its targets.
  const away = level(soma.clone().sub(toward));
  const look = soma.clone().lerp(toward, 0.35);
  const view = (dir: Vector3, d: number) => pose(soma.clone().addScaledVector(dir, d).addScaledVector(WORLD_UP, d * 0.3), look.clone());
  const fire = 7;
  // Slow motion while it lands: at normal speed a landing glows for about a tenth of a second.
  const slow = 0.2;
  const landed = fire + landsAfter(farthestUm, c) / slow + 0.3;
  return {
    id: "basket",
    title: "A basket cell silences its neighbours",
    blurb: `One inhibitory cell and the ${formatCount(pick.targets)} cells it reaches.`,
    neuron: pick.neuron,
    seconds: Math.max(26, landed + 14),
    steps: byTime([
      { at: 0, do: { kind: "fly", pose: view(away, 170), seconds: 5 } },
      {
        at: 0.8,
        do: {
          kind: "caption",
          text: `This basket cell makes ${formatCount(pick.synapses)} synapses onto ${formatCount(pick.targets)} neighbouring cells, and every one of them is inhibitory.`,
          seconds: 5.6,
        },
      },
      // A slow drift round the soma for the rest of the story.
      { at: 5, do: { kind: "fly", pose: view(turn(away, 0.5), 150), seconds: 25 } },
      { at: fire - 0.4, do: { kind: "speed", scale: slow } },
      { at: fire, do: { kind: "stimulate", neuron: pick.neuron } },
      {
        at: fire + 0.3,
        do: {
          kind: "caption",
          text: `It fired. Slowed ${formatCount(1 / slow)} times more, each violet point is one of its synapses as the pulse lands.`,
          seconds: landed - fire - 0.5,
        },
      },
      {
        at: landed,
        do: { kind: "caption", text: () => `${formatCount(live.reached())} cells just received a pulse that pushes them further from firing.`, seconds: 5.5 },
      },
      {
        at: landed + 6,
        do: {
          kind: "caption",
          text: "Inhibition is a spike that does not happen, so it is drawn: violet where each pulse lands, and a brief dimming of every cell it reached.",
          seconds: 7,
        },
      },
      { at: landed + 7, do: { kind: "speed", scale: 1 } },
    ]),
  };
}

function axonStory(data: Dataset, c: StoryConduction): StoryDef | null {
  const pick = pickLongestAxon(data);
  if (!pick) return null;
  const soma = somaOf(data, pick.neuron);
  const track = axonTrack(data, pick.neuron)!;
  const lo = new Vector3(Infinity, Infinity, Infinity);
  const hi = new Vector3(-Infinity, -Infinity, -Infinity);
  const p = new Vector3();
  for (let i = 0; i < track.count; i++) {
    p.fromArray(track.xyz, i * 3);
    lo.min(p);
    hi.max(p);
  }
  // Near: behind the soma, looking out along where the axon leaves it.
  const out = level(trackPoint(track, Math.min(40, track.s[track.count - 1]), new Vector3()).sub(soma));
  const near = pose(soma.clone().addScaledVector(out, -75).addScaledVector(WORLD_UP, 25), soma.clone().addScaledVector(out, 60));
  // Wide: side on to the branch's box, looking along its thinner horizontal axis, far enough back to hold all of it.
  const centre = lo.clone().add(hi).multiplyScalar(0.5);
  const size = hi.clone().sub(lo);
  const xThin = size.x < size.z;
  const along = xThin ? new Vector3(1, 0, 0) : new Vector3(0, 0, 1);
  const lateral = xThin ? size.z : size.x;
  const depth = xThin ? size.x : size.z;
  const back = (1.1 * Math.max(size.y / 2, lateral / 2 / 1.6)) / Math.tan(HALF_FOV) + depth / 2;
  const side = centre.clone().sub(soma).dot(along) > 0 ? -1 : 1;
  const wide = (radians: number) =>
    pose(centre.clone().addScaledVector(turn(along, radians), side * back).addScaledVector(WORLD_UP, back * 0.12), centre.clone());

  const shownS = landsAfter(pick.lengthUm, c) - c.synDelayS;
  const refire = 10.5;
  const info = describeNeuron(data, pick.neuron);
  const typeName = cellNoun(info.typeName);
  const travelled = refire + 0.3 + Math.max(6, shownS + 1.5);
  return {
    id: "axon",
    title: "The longest axon in the volume",
    blurb: `Ride one spike down ${formatLength(pick.lengthUm)} of wire.`,
    neuron: pick.neuron,
    seconds: Math.min(45, Math.max(26, travelled + 7)),
    steps: [
      { at: 0, do: { kind: "fly", pose: near, seconds: 4.5 } },
      {
        at: 0.8,
        do: {
          kind: "caption",
          text: `The longest axon in the volume belongs to this ${typeName}: ${formatLength(pick.lengthUm)} from its soma to its farthest tip.`,
          seconds: 5.2,
        },
      },
      { at: 6, do: { kind: "ride", neuron: pick.neuron } },
      {
        at: 6.1,
        // Stays up through the ride (the timeline holds) and the pull back, until the second firing.
        do: { kind: "caption", text: `Riding its spike. The pulse runs at ${c.conductionMps} m/s, shown ${formatCount(c.slowMo)} times slower.`, seconds: refire - 6.1 },
      },
      // Hold while the ride camera follows the pulse, until it jumps into the cell the pulse reaches.
      { at: 6.2, do: { kind: "wait", event: "rideJump", timeout: Math.min(12, shownS + 3) } },
      { at: 6.3, do: { kind: "fly", pose: wide(0), seconds: 3.5 } },
      { at: 9.8, do: { kind: "fly", pose: wide(0.35), seconds: 30 } },
      { at: refire, do: { kind: "stimulate", neuron: pick.neuron } },
      {
        at: refire + 0.3,
        do: {
          kind: "caption",
          text: `One spike, ${formatLength(pick.lengthUm)} of wire, ${shownS.toFixed(1)} seconds at ${formatCount(c.slowMo)}x slower than life.`,
          seconds: Math.max(6, shownS + 1.5),
        },
      },
      ...(info.outgoing
        ? [
            {
              at: travelled + 0.5,
              do: {
                kind: "caption",
                text: `Along the way it makes ${formatCount(info.outgoing)} synapses onto other cells in this volume.`,
                seconds: 5.5,
              },
            } satisfies StoryStep,
          ]
        : []),
    ],
  };
}

function layersStory(data: Dataset, live: StoryLive, c: StoryConduction): StoryDef | null {
  const pick = pickFeedforward(data);
  if (!pick) return null;
  const soma = somaOf(data, pick.neuron);
  const { cells, farthestUm } = targetsOf(data, pick.neuron, (post) => typeOf(data, post) === "23P");
  const up = medianSoma(data, cells);
  // Frames layers 2/3 and 4 between their fitted boundaries (pia at the top of the screen), centred on
  // the cell and its targets, seen from the nearer z face; without a layer fit, the cell and its targets.
  const bounds = layerBoundaries(data);
  const top = bounds.length === 6 ? bounds[1] : Math.min(soma.y, up.y) - 40;
  const bottom = bounds.length === 6 ? bounds[4] : Math.max(soma.y, up.y) + 40;
  const mid = soma.clone().lerp(up, 0.5).setY((top + bottom) / 2);
  const { min, max } = data.manifest.boundsUm;
  const face = new Vector3(0, 0, mid.z - min[2] < max[2] - mid.z ? -1 : 1);
  const d = ((bottom - top) / 2 + 30) / Math.tan(HALF_FOV);
  const view = (radians: number) => pose(mid.clone().addScaledVector(turn(face, radians), d).addScaledVector(WORLD_UP, d * 0.12), mid.clone());
  const fire = 7.5;
  const result = fire + Math.max(1.5, landsAfter(farthestUm, c) + 0.5);
  return {
    id: "layers",
    title: "Layer 4 feeds layer 2/3",
    blurb: "The cortex's first feedforward step, fired once.",
    neuron: pick.neuron,
    seconds: Math.max(26, result + 11),
    steps: byTime([
      { at: 0, do: { kind: "fly", pose: view(-0.25), seconds: 5 } },
      {
        at: 0.8,
        do: {
          kind: "caption",
          text: `Layer 4 is the cortex's input layer. This cell sends ${formatCount(pick.synapses)} synapses up to ${formatCount(pick.targets)} layer 2/3 cells.`,
          seconds: 4.8,
        },
      },
      { at: 5, do: { kind: "fly", pose: view(0.2), seconds: 22 } },
      { at: fire, do: { kind: "stimulate", neuron: pick.neuron } },
      {
        at: fire - 1.7,
        do: { kind: "caption", text: "When it fires, each bright line will run from it to a cell its pulse sets off.", seconds: 2.6 },
      },
      {
        at: fire + 1,
        do: {
          kind: "caption",
          text: () => `${formatCount(live.cascade().hop1Cells.filter((n) => cells.has(n)).length)} of those ${formatCount(cells.size)} layer 2/3 cells fired.`,
          seconds: result + 4 - (fire + 1),
          live: true,
        },
      },
      {
        at: result + 4.5,
        do: { kind: "caption", text: () => `Counting the cells those set off, ${formatCount(live.cascade().total)} fired in all.`, seconds: 5.5 },
      },
    ]),
  };
}

function hubStory(data: Dataset): StoryDef | null {
  const pick = pickHub(data);
  if (!pick) return null;
  const info = describeNeuron(data, pick.neuron);
  const orbit: OrbitAction = { kind: "orbit", centre: somaOf(data, pick.neuron), radiusUm: 170, liftUm: 45, fromAngle: 0, seconds: 15 };
  const start = 5;
  const end = start + orbit.seconds;
  const blank = () => pose(new Vector3(), new Vector3());
  // After the orbit, ease a little further back from where it began.
  const settle = orbitPose(orbit, 0, blank());
  settle.position.lerp(orbit.centre, -0.25);
  return {
    id: "hub",
    title: "One cell's whole world",
    blurb: `Circle the most connected cell and its ${formatCount(pick.partners)} partners.`,
    neuron: pick.neuron,
    seconds: end + 6,
    steps: [
      { at: 0, do: { kind: "fly", pose: orbitPose(orbit, 0, blank()), seconds: start } },
      {
        at: 0.8,
        do: {
          kind: "caption",
          text: `This ${cellNoun(info.typeName)} talks to ${formatCount(pick.partners)} other cells: ${formatCount(pick.inputs)} synapses in, ${formatCount(pick.outputs)} out, reaching cells up to ${formatLength(pick.reachUm)} away.`,
          seconds: 6,
        },
      },
      { at: start, do: { kind: "partners", neuron: pick.neuron, show: true } },
      { at: start, do: orbit },
      {
        at: 7.5,
        do: { kind: "caption", text: `By synapses in and out, it is the most connected of the ${formatCount(data.neurons.count)} cells here.`, seconds: 5.5 },
      },
      { at: 11, do: { kind: "stimulate", neuron: pick.neuron } },
      {
        at: 13.5,
        do: {
          kind: "caption",
          text:
            info.className === "inhibitory"
              ? `It is inhibitory: each of its ${formatCount(pick.outputs)} output synapses pushes its target away from firing.`
              : `It is excitatory: each of its ${formatCount(pick.outputs)} output synapses nudges its target toward firing.`,
          seconds: 6,
        },
      },
      { at: end, do: { kind: "partners", neuron: pick.neuron, show: false } },
      { at: end, do: { kind: "fly", pose: settle, seconds: 5 } },
    ],
  };
}

/** The stories this dataset can tell, in menu order. */
export function buildStories(data: Dataset, live: StoryLive, conduction: StoryConduction = DEFAULT_CONDUCTION): StoryDef[] {
  return [basketStory(data, live, conduction), axonStory(data, conduction), layersStory(data, live, conduction), hubStory(data)].filter(
    (s): s is StoryDef => s !== null,
  );
}
