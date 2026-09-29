/**
 * The intro tour: a ~20 s scripted first look at the volume, played once per visitor.
 *
 * Everything above `Tour` is pure and tested without a GPU: when the tour runs, the poses it
 * moves between, and which caption shows at a given time. `Tour` drives the shared `Flight`
 * through the timeline; it never writes the camera itself.
 */

import { Vector3, type PerspectiveCamera } from "three/webgpu";
import { WORLD_UP, heroPath, type HeroAnchor } from "./camera/fly";
import type { Flight, Pose } from "./camera/goto";
import type { EventBus } from "./events";
import { DEFAULT_CONDUCTION_MPS, DEFAULT_SLOW_MO } from "./camera/ride";

export interface TourGate {
  /** The `cmm-tour` pref says this visitor has seen it. */
  seen: boolean;
  shot: boolean;
  parity: boolean;
  /** `location.hash`; a camera link (`#c=...`) opens on its own view instead. */
  hash: string;
  /** `?tour=1`. */
  forced: boolean;
}

export function shouldRunTour({ seen, shot, parity, hash, forced }: TourGate): boolean {
  if (shot || parity) return false;
  if (forced) return true;
  return !seen && !hash.replace(/^#/, "").startsWith("c=");
}

/** Keyframes, seconds from the start of the tour. */
export const TOUR = {
  /** Drift from the wide opening view in to the hold beside the hero. */
  driftEnd: 6,
  /** The hero is stimulated here, as the drift settles. */
  stimulus: 6,
  /** The camera holds while the cascade spreads, then orbits round to the resting pose. */
  orbitStart: 13,
  orbitEnd: 18,
  /** A skip flies straight to the resting pose over this long. */
  skipFlight: 1,
} as const;
export const TOUR_SECONDS = 20;

/** The opening is dim and fades up to full brightness over this many tour seconds. */
export const INTRO_FADE_SECONDS = 5;
/** Brightness at tour time 0. */
export const INTRO_FADE_FROM = 0.35;

/** Scene brightness (the `introFade` uniform) at tour time `t`: a dark vast space that lights as you fall in. */
export function introFade(t: number): number {
  const k = Math.max(0, Math.min(1, t / INTRO_FADE_SECONDS));
  return INTRO_FADE_FROM + (1 - INTRO_FADE_FROM) * k * k * (3 - 2 * k);
}

/** When each caption is on screen, [from, to) in tour seconds. The HUD fades them in and out. */
export const CAPTION_WINDOWS: readonly (readonly [number, number])[] = [
  [0.8, 5.6],
  [7.4, 12.8],
  [15.4, 19.6],
];

/** Index of the caption on screen at tour time `t`, -1 for none. */
export function captionAt(t: number): number {
  return CAPTION_WINDOWS.findIndex(([from, to]) => t >= from && t < to);
}

/** The three caption lines. `depthUm` is the hero soma's depth below the pia (`describeNeuron`). */
export function tourCaptions(depthUm: number, coarse: boolean): [string, string, string] {
  return [
    `You are ${Math.round(depthUm / 10) * 10} µm below the surface of a mouse's visual cortex. Every cell here is real.`,
    `That was one spike, travelling down a real axon at ${DEFAULT_CONDUCTION_MPS} m/s, slowed ${DEFAULT_SLOW_MO} times.`,
    coarse ? "Tap a cell. Left thumb moves, right thumb looks." : "Click a cell. Space fires it. R rides the spike.",
  ];
}

export interface TourPoses {
  /** Where the tour opens: back and to the side of the cluster, high, looking at its centre, so structure fills the view. */
  wide: Pose;
  /** Behind the hero, looking out along its axon: close enough to watch its spike and the cascade after it. */
  hold: Pose;
  /** The resting pose: the view a visitor starts from, and where Home returns to. */
  rest: Pose;
}

/** Offsets in the hero's own frame (µm along its axon, to its side, and up toward the pia). */
function framePose(anchor: HeroAnchor, eye: [number, number, number], look: [number, number, number]): Pose {
  const side = new Vector3().crossVectors(anchor.axis, WORLD_UP).normalize();
  const up = new Vector3().crossVectors(side, anchor.axis).normalize();
  const at = ([a, s, u]: [number, number, number]) =>
    anchor.soma.clone().addScaledVector(anchor.axis, a).addScaledVector(side, s).addScaledVector(up, u);
  return { position: at(eye), target: at(look) };
}

export function tourPoses(anchor: HeroAnchor): TourPoses {
  return {
    // Just outside the cluster, still framing it, looking at its centre rather than past the hero.
    wide: framePose(anchor, [-260, 120, 80], [0, 0, 0]),
    // The resting composition mirrored across the axon: the spike runs away from the camera into the cluster.
    hold: framePose(anchor, [-105, -95, 38], [190, -10, -10]),
    rest: heroPath(0, anchor, { position: new Vector3(), target: new Vector3() }),
  };
}

export interface TourOptions {
  /** The app's one shared flight; the tour starts its legs, the app loop advances it. */
  flight: Flight;
  camera: PerspectiveCamera;
  bus: EventBus;
  heroNeuron: number;
  anchor: HeroAnchor;
  /** Called once each time the tour ends or is skipped (the app marks the pref seen). */
  onEnd?: () => void;
}

/**
 * Plays the timeline on real (wall-clock) seconds: the app calls `update(dt)` every frame, before
 * `flight.update(dt)`. Emits `tour {running}` at start and end, and `stimulate` for the hero at
 * `TOUR.stimulus`.
 */
export class Tour {
  readonly poses: TourPoses;
  private t = 0;
  private active = false;

  constructor(private readonly opts: TourOptions) {
    this.poses = tourPoses(opts.anchor);
  }

  get running(): boolean {
    return this.active;
  }

  /** Seconds since the tour started. */
  get time(): number {
    return this.t;
  }

  /** Cuts to the wide opening view and starts the timeline from 0. */
  start(): void {
    const { camera, flight, bus } = this.opts;
    const { wide, hold } = this.poses;
    this.t = 0;
    this.active = true;
    camera.position.copy(wide.position);
    camera.up.copy(WORLD_UP);
    camera.lookAt(wide.target);
    flight.start(hold, TOUR.driftEnd);
    bus.emit("tour", { running: true });
  }

  update(dt: number): void {
    if (!this.active) return;
    const prev = this.t;
    this.t += dt;
    const crossed = (cue: number) => prev < cue && this.t >= cue;
    if (crossed(TOUR.stimulus)) this.opts.bus.emit("stimulate", { neuron: this.opts.heroNeuron });
    if (crossed(TOUR.orbitStart)) this.opts.flight.start(this.poses.rest, TOUR.orbitEnd - TOUR.orbitStart);
    if (this.t >= TOUR_SECONDS) this.end();
  }

  /** Ends the tour now and flies to the resting pose. A no-op when it is not running. */
  skip(): void {
    if (!this.active) return;
    this.opts.flight.start(this.poses.rest, TOUR.skipFlight);
    this.end();
  }

  private end(): void {
    this.active = false;
    this.opts.bus.emit("tour", { running: false });
    this.opts.onEnd?.();
  }
}
