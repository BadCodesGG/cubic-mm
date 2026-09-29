/**
 * Ride the spike. After `ride {neuron}` the camera waits for that neuron's next spike, then rides
 * the pulse front down its longest axon branch, jumps into the postsynaptic cell at the synapse
 * the front reaches, and follows the signal through that dendrite toward its soma before easing
 * back out and handing the camera to `FlyControls`.
 *
 * The pulse front is at path distance
 *
 *     d(t) = (t - spikeTime) * conductionMps * 1e6 / slowMo         [µm]
 *
 * (m/s to µm/s is x 1e6; slowMo is the display slow-down the shader applies to conduction delay).
 * That is the same law the neuron shader uses, so the camera sits on the glowing front.
 *
 * Everything before `RideCamera` is pure and tested without a GPU: track extraction, the point
 * at a distance along a track, the front law, and the choice of the synapse to jump at.
 */

import { PerspectiveCamera, Vector3 } from "three/webgpu";
import { Compartment, NO_PARENT, PATH_DIST_UNIT_UM, dequantise } from "../format";
import type { Dataset } from "../data";
import type { EventBus } from "../events";
import { WORLD_UP, type FlyControls } from "./fly";

export const DEFAULT_CONDUCTION_MPS = 0.5;
export const DEFAULT_SLOW_MO = 1000;
/** The camera rides this far behind the front, looking this far ahead of it, µm. */
export const BEHIND_UM = 6;
export const AHEAD_UM = 15;
/** Shortest stretch of axon ridden before a synapse is accepted as the jump point, µm. */
export const MIN_RIDE_UM = 120;
/** A synapse belongs to the ridden branch if it is within this many µm of the branch at its path distance. */
const BRANCH_TOLERANCE_UM = 6;
export const EASE_OUT_SECONDS = 1.5;

/** A polyline with a monotonically increasing ride distance `s` per vertex (µm). */
export interface Track {
  count: number;
  /** xyz per vertex, µm. */
  xyz: Float32Array;
  /** Ride distance of each vertex, µm, non-decreasing. */
  s: Float32Array;
}

/** Path distance the pulse front has covered `t - spikeTime` seconds after the spike, µm. */
export function frontDistanceUm(t: number, spikeTime: number, conductionMps: number, slowMo: number): number {
  return ((t - spikeTime) * conductionMps * 1e6) / slowMo;
}

function nodePosition(data: Dataset, node: number, out: Float32Array, at: number): void {
  const { min, max } = data.manifest.boundsUm;
  for (let k = 0; k < 3; k++) out[at + k] = dequantise(data.nodes.pos[node * 3 + k], min[k], max[k]);
}

function trackFromChain(data: Dataset, chain: number[], distance: (node: number) => number): Track {
  const xyz = new Float32Array(chain.length * 3);
  const s = new Float32Array(chain.length);
  let prev = 0;
  chain.forEach((node, i) => {
    nodePosition(data, node, xyz, i * 3);
    prev = Math.max(prev, distance(node));
    s[i] = prev;
  });
  return { count: chain.length, xyz, s };
}

/** Nodes from `start` up through `parent` links to the root, `start` first. */
function chainToRoot(data: Dataset, start: number, limit: number): number[] {
  const chain: number[] = [];
  for (let node = start; node !== NO_PARENT && chain.length <= limit; node = data.nodes.parent[node]) chain.push(node);
  return chain;
}

/**
 * The longest axon branch of a neuron: from the soma to the axon node farthest away along the
 * tree (largest path distance), one vertex per skeleton node. `s` is the stored path distance from
 * the soma, so it agrees with `preDistQ` in the synapse table. Null if the neuron has no axon.
 */
export function axonTrack(data: Dataset, neuron: number): Track | null {
  const start = data.neuronNodeStart[neuron];
  const end = start + data.neuronNodeCount[neuron];
  let tip = -1;
  let best = -1;
  for (let i = start; i < end; i++) {
    if (data.nodes.compartment[i] !== Compartment.Axon) continue;
    if (data.nodes.pathDistQ[i] > best) {
      best = data.nodes.pathDistQ[i];
      tip = i;
    }
  }
  if (tip < 0) return null;
  const chain = chainToRoot(data, tip, data.neuronNodeCount[neuron]).reverse();
  return trackFromChain(data, chain, (n) => data.nodes.pathDistQ[n] * PATH_DIST_UNIT_UM);
}

/**
 * From `postNode` toward the soma of the same neuron, walking `parent` links, so the stored path
 * distance falls while the ride distance `s` (path distance from `postNode`) rises.
 */
export function towardSomaTrack(data: Dataset, postNode: number): Track {
  const neuron = data.nodes.neuronOfNode[postNode];
  const chain = chainToRoot(data, postNode, data.neuronNodeCount[neuron]);
  const origin = data.nodes.pathDistQ[postNode] * PATH_DIST_UNIT_UM;
  return trackFromChain(data, chain, (n) => origin - data.nodes.pathDistQ[n] * PATH_DIST_UNIT_UM);
}

/** Unit direction of the first (or last) non-degenerate segment of a track, for extrapolating past its ends. */
function endDirection(track: Track, atEnd: boolean, out: Vector3): Vector3 {
  const { xyz, count } = track;
  const a = atEnd ? count - 1 : 0;
  for (let k = 1; k < count; k++) {
    const b = atEnd ? a - k : k;
    out.set(xyz[a * 3] - xyz[b * 3], xyz[a * 3 + 1] - xyz[b * 3 + 1], xyz[a * 3 + 2] - xyz[b * 3 + 2]);
    if (out.lengthSq() > 1e-9) {
      out.normalize();
      return atEnd ? out : out.negate();
    }
  }
  return out.set(1, 0, 0);
}

const dir = new Vector3();

/**
 * The point at ride distance `s`. Inside the track it interpolates along the segment whose `s`
 * range holds `s`; past either end it continues straight along the end segment.
 */
export function trackPoint(track: Track, s: number, out: Vector3): Vector3 {
  const { xyz, s: ss, count } = track;
  if (count === 1) return out.set(xyz[0], xyz[1], xyz[2]);
  if (s <= ss[0]) {
    endDirection(track, false, dir);
    return out.set(xyz[0], xyz[1], xyz[2]).addScaledVector(dir, s - ss[0]);
  }
  const last = count - 1;
  if (s >= ss[last]) {
    endDirection(track, true, dir);
    return out.set(xyz[last * 3], xyz[last * 3 + 1], xyz[last * 3 + 2]).addScaledVector(dir, s - ss[last]);
  }
  // Largest i with ss[i] <= s.
  let lo = 0;
  let hi = last;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (ss[mid] <= s) lo = mid;
    else hi = mid;
  }
  const span = ss[hi] - ss[lo];
  const f = span > 1e-9 ? (s - ss[lo]) / span : 0;
  return out.set(
    xyz[lo * 3] + (xyz[hi * 3] - xyz[lo * 3]) * f,
    xyz[lo * 3 + 1] + (xyz[hi * 3 + 1] - xyz[lo * 3 + 1]) * f,
    xyz[lo * 3 + 2] + (xyz[hi * 3 + 2] - xyz[lo * 3 + 2]) * f,
  );
}

export interface JumpTarget {
  synapse: number;
  /** Ride distance of the synapse along the presynaptic track, µm. */
  s: number;
  post: number;
  /** Node of the postsynaptic neuron nearest the synapse: where the ride resumes. */
  postNode: number;
}

const probe = new Vector3();
const synPos = new Vector3();

/**
 * The synapse to jump at: the one with the smallest `preDistQ` at or beyond `minS` that lies on
 * the ridden branch. Null when the dataset has no synapse table or none qualifies, in which case
 * the ride runs to the branch tip.
 */
export function chooseSynapse(data: Dataset, neuron: number, track: Track, minS: number): JumpTarget | null {
  const syn = data.synapses;
  if (!syn) return null;
  const { min, max } = data.manifest.boundsUm;
  const endS = track.s[track.count - 1];
  let best: JumpTarget | null = null;
  for (let i = syn.preOffsets[neuron]; i < syn.preOffsets[neuron + 1]; i++) {
    const s = syn.preDistQ[i] * PATH_DIST_UNIT_UM;
    if (s < minS || s > endS || (best && s >= best.s)) continue;
    const x = dequantise(syn.pos[i * 3], min[0], max[0]);
    const y = dequantise(syn.pos[i * 3 + 1], min[1], max[1]);
    const z = dequantise(syn.pos[i * 3 + 2], min[2], max[2]);
    if (trackPoint(track, s, probe).distanceTo(synPos.set(x, y, z)) > BRANCH_TOLERANCE_UM) continue;
    best = { synapse: i, s, post: syn.post[i], postNode: nearestNode(data, syn.post[i], x, y, z) };
  }
  return best;
}

/** Node of `neuron` closest to a point, µm. */
export function nearestNode(data: Dataset, neuron: number, x: number, y: number, z: number): number {
  const { min, max } = data.manifest.boundsUm;
  const start = data.neuronNodeStart[neuron];
  const end = start + data.neuronNodeCount[neuron];
  let best = start;
  let bestD = Infinity;
  for (let i = start; i < end; i++) {
    const dx = dequantise(data.nodes.pos[i * 3], min[0], max[0]) - x;
    const dy = dequantise(data.nodes.pos[i * 3 + 1], min[1], max[1]) - y;
    const dz = dequantise(data.nodes.pos[i * 3 + 2], min[2], max[2]) - z;
    const d = dx * dx + dy * dy + dz * dz;
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

/** Critically damped spring toward `target` (the SmoothDamp form): no overshoot, reaches it in about 4 x smoothTime. */
export function smoothDamp(current: number, target: number, velocity: { v: number }, smoothTime: number, dt: number): number {
  const omega = 2 / Math.max(1e-4, smoothTime);
  const x = omega * dt;
  const decay = 1 / (1 + x + 0.48 * x * x + 0.235 * x * x * x);
  const change = current - target;
  const temp = (velocity.v + omega * change) * dt;
  velocity.v = (velocity.v - omega * temp) * decay;
  return target + (change + temp) * decay;
}

export type RideState = "idle" | "waiting" | "riding" | "easing";

export interface ConductionParams {
  conductionMps: number;
  slowMo: number;
}

/** Reads `window.__cmm.sim.params` when the simulation publishes it, else the documented defaults. */
export function readConduction(): ConductionParams {
  const sim = (typeof window !== "undefined" ? (window.__cmm as { sim?: { params?: Partial<ConductionParams> } } | undefined)?.sim : undefined);
  const p = sim?.params;
  return {
    conductionMps: p?.conductionMps && p.conductionMps > 0 ? p.conductionMps : DEFAULT_CONDUCTION_MPS,
    slowMo: p?.slowMo && p.slowMo > 0 ? p.slowMo : DEFAULT_SLOW_MO,
  };
}

interface Vec3Damper {
  pos: Vector3;
  vx: { v: number };
  vy: { v: number };
  vz: { v: number };
}

function damper(): Vec3Damper {
  return { pos: new Vector3(), vx: { v: 0 }, vy: { v: 0 }, vz: { v: 0 } };
}

function dampVec(d: Vec3Damper, goal: Vector3, smoothTime: number, dt: number): void {
  d.pos.set(
    smoothDamp(d.pos.x, goal.x, d.vx, smoothTime, dt),
    smoothDamp(d.pos.y, goal.y, d.vy, smoothTime, dt),
    smoothDamp(d.pos.z, goal.z, d.vz, smoothTime, dt),
  );
}

export class RideCamera {
  state: RideState = "idle";
  /** The neuron being waited on, then the one being ridden. */
  neuron = -1;

  private readonly tracks = new Map<number, Track | null>();
  private readonly offs: (() => void)[] = [];
  private readonly eye = damper();
  private readonly gaze = damper();
  private readonly goalEye = new Vector3();
  private readonly goalGaze = new Vector3();
  private readonly forward = new Vector3();
  private readonly prevForward = new Vector3();
  private readonly tangent = new Vector3();
  private readonly roll = { v: 0 };
  private rollAngle = 0;

  private spikeTime = 0;
  private speed = 0;
  private track: Track | null = null;
  private jump: JumpTarget | null = null;
  private jumpTime = 0;
  private onDendrite = false;
  private elapsed = 0;
  private easeElapsed = 0;
  private endS = 0;
  private wasLocked = false;
  private readonly conduction: () => ConductionParams;
  private readonly minRideUm: number;

  constructor(
    private readonly data: Dataset,
    private readonly bus: EventBus,
    private readonly camera: PerspectiveCamera,
    private readonly controls: FlyControls | null,
    options: { conduction?: () => ConductionParams; minRideUm?: number } = {},
  ) {
    this.conduction = options.conduction ?? readConduction;
    this.minRideUm = options.minRideUm ?? MIN_RIDE_UM;
    this.offs.push(
      bus.on("ride", (e) => this.request(e.neuron)),
      bus.on("spike", (e) => {
        if (this.state === "waiting" && e.neuron === this.neuron) this.begin(e.time);
      }),
      bus.on("select", (e) => {
        if (this.state === "waiting" && e.neuron !== this.neuron) this.cancel();
      }),
    );
    window.addEventListener("keydown", this.onKeyDown);
    document.addEventListener("pointerlockchange", this.onLockChange);
  }

  dispose(): void {
    this.cancel();
    for (const off of this.offs) off();
    window.removeEventListener("keydown", this.onKeyDown);
    document.removeEventListener("pointerlockchange", this.onLockChange);
  }

  /** True while the ride owns the camera. */
  get active(): boolean {
    return this.state === "riding" || this.state === "easing";
  }

  /** Stop waiting or riding and give the camera back where it is. */
  cancel(): void {
    const wasActive = this.active;
    this.state = "idle";
    this.track = null;
    this.jump = null;
    if (wasActive) this.handBack();
  }

  private request(neuron: number): void {
    if (!(neuron >= 0 && neuron < this.data.neurons.count)) return;
    if (this.active) this.cancel();
    this.neuron = neuron;
    this.state = "waiting";
  }

  private axon(neuron: number): Track | null {
    if (!this.tracks.has(neuron)) this.tracks.set(neuron, axonTrack(this.data, neuron));
    return this.tracks.get(neuron) ?? null;
  }

  private begin(spikeTime: number): void {
    const track = this.axon(this.neuron);
    if (!track) {
      this.state = "idle";
      return;
    }
    const { conductionMps, slowMo } = this.conduction();
    this.speed = (conductionMps * 1e6) / slowMo;
    this.spikeTime = spikeTime;
    this.track = track;
    this.jump = chooseSynapse(this.data, this.neuron, track, Math.min(this.minRideUm, track.s[track.count - 1]));
    this.endS = this.jump ? this.jump.s : track.s[track.count - 1];
    this.onDendrite = false;
    this.elapsed = 0;
    this.easeElapsed = 0;
    this.wasLocked = document.pointerLockElement != null;
    this.state = "riding";
    if (this.controls) this.controls.enabled = false;

    // Start the springs from where the visitor's camera is now, so the ride glides in.
    this.camera.getWorldDirection(this.forward);
    this.eye.pos.copy(this.camera.position);
    this.gaze.pos.copy(this.camera.position).addScaledVector(this.forward, AHEAD_UM);
    for (const v of [this.eye.vx, this.eye.vy, this.eye.vz, this.gaze.vx, this.gaze.vy, this.gaze.vz, this.roll]) v.v = 0;
    this.rollAngle = 0;
    this.prevForward.copy(this.forward);
  }

  /** Advance the ride to simulation time `simTime`. Returns true if it set the camera this frame. */
  update(simTime: number, dt: number): boolean {
    if (!this.active || !this.track) return false;
    this.elapsed += dt;

    if (this.state === "riding") {
      if (!this.onDendrite && this.rawS(simTime) >= this.endS) this.reachEnd();
      if (this.onDendrite && this.state === "riding" && this.rawS(simTime) >= this.endS) this.beginEase();
    }
    const s = this.state === "easing" ? this.endS : Math.min(this.endS, this.rawS(simTime));
    if (this.state === "easing") this.easeElapsed += dt;

    const track = this.track!;
    // Looser while gliding in from wherever the visitor was, tighter once on the pulse.
    const smooth = 0.3 + (0.14 - 0.3) * Math.min(1, this.elapsed);
    // A spring following a target that moves at `speed` trails it by speed * smoothTime, so the goal
    // leads by that much: the camera then sits on "6 µm behind the front" instead of tens of µm back.
    // The lead fades out over the ease, so the camera stops without jerking.
    const k = this.state === "easing" ? Math.min(1, this.easeElapsed / EASE_OUT_SECONDS) : 0;
    const lead = this.speed * smooth * (1 - k);
    if (this.state === "easing") {
      // Stopped at the end of the track, backing off a little to show the cell.
      const back = BEHIND_UM + 22 * (1 - (1 - k) * (1 - k));
      trackPoint(track, s - back + lead, this.goalEye);
      trackPoint(track, s, this.goalGaze);
    } else {
      trackPoint(track, s - BEHIND_UM + lead, this.goalEye);
      trackPoint(track, s + AHEAD_UM + lead, this.goalGaze);
    }
    dampVec(this.eye, this.goalEye, smooth, dt);
    dampVec(this.gaze, this.goalGaze, smooth, dt);

    this.camera.position.copy(this.eye.pos);
    this.forward.copy(this.gaze.pos).sub(this.eye.pos);
    if (this.forward.lengthSq() < 1e-8) this.forward.copy(this.prevForward);
    this.forward.normalize();

    // A slight roll into turns: proportional to how fast the heading swings about the world up axis.
    const swing = this.tangent.crossVectors(this.prevForward, this.forward).dot(WORLD_UP) / Math.max(dt, 1e-3);
    this.prevForward.copy(this.forward);
    this.rollAngle = smoothDamp(this.rollAngle, Math.max(-0.14, Math.min(0.14, -swing * 0.12)), this.roll, 0.25, dt);
    this.camera.up.copy(WORLD_UP).applyAxisAngle(this.forward, this.rollAngle);
    this.camera.lookAt(this.gaze.pos);

    if (this.state === "easing" && this.easeElapsed >= EASE_OUT_SECONDS) this.cancel();
    return true;
  }

  /** Ride distance of the pulse front on the current track, unclamped. */
  private rawS(simTime: number): number {
    return this.speed * (simTime - (this.onDendrite ? this.jumpTime : this.spikeTime));
  }

  /** The front reached the synapse (or the tip): jump into the post cell, or ease out. */
  private reachEnd(): void {
    const jump = this.jump;
    if (!jump) {
      this.beginEase();
      return;
    }
    const post = towardSomaTrack(this.data, jump.postNode);
    this.jumpTime = this.spikeTime + jump.s / this.speed;
    this.bus.emit("rideJump", { fromNeuron: this.neuron, neuron: jump.post, time: this.jumpTime });
    this.neuron = jump.post;
    this.track = post;
    this.jump = null;
    this.onDendrite = true;
    this.endS = post.s[post.count - 1];
    if (this.endS < 1e-6) this.beginEase();
  }

  private beginEase(): void {
    this.state = "easing";
    this.easeElapsed = 0;
  }

  private handBack(): void {
    this.camera.up.copy(WORLD_UP);
    if (!this.controls) return;
    this.controls.lookAt(this.gaze.pos);
    this.controls.enabled = true;
  }

  private onKeyDown = (e: KeyboardEvent) => {
    if (e.code !== "Escape" || this.state === "idle") return;
    e.preventDefault();
    this.cancel();
  };

  private onLockChange = () => {
    if (this.active && this.wasLocked && document.pointerLockElement == null) this.cancel();
  };
}
