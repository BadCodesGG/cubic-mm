/**
 * Render quality: which skeleton LOD the ribbons are built from ("hi" or "lite") and how many
 * curve pieces each skeleton edge is split into (1 to 3; the instance count scales with it).
 *
 * `resolveQuality` picks the starting level from the saved preference and the device. On "auto"
 * the level is adaptive: `FrameBudget` watches the frame time after start-up and, while it is over
 * budget, the app rebuilds one rung further down `LADDER`. It never steps back up by itself.
 */

import type { Quality } from "./prefs";

export type Tier = "hi" | "lite";
export type Pieces = 1 | 2 | 3;

export interface QualityLevel {
  tier: Tier;
  pieces: Pieces;
}

export interface QualityInputs {
  /** The `cmm-quality` preference. */
  pref: Quality;
  isWebGPU: boolean;
  /** `(pointer: coarse)` matches: a phone or tablet. */
  coarsePointer: boolean;
  viewportWidth: number;
  /** `?quality=` / `?pieces=` from the URL; pins the level and turns adapting off. */
  override?: Partial<QualityLevel>;
  /** Screenshot and parity runs must render the same frame every time, so they never adapt. */
  scripted?: boolean;
}

export interface QualityPlan {
  level: QualityLevel;
  /** Whether the frame-time check may step the level down. */
  adaptive: boolean;
}

/** Viewports narrower than this, in CSS px, start on lite. */
export const NARROW_PX = 900;

export function resolveQuality(i: QualityInputs): QualityPlan {
  let level: QualityLevel;
  if (i.pref === "hi") level = { tier: "hi", pieces: 3 };
  else if (i.pref === "lite" || !i.isWebGPU || i.coarsePointer || i.viewportWidth < NARROW_PX) level = { tier: "lite", pieces: 2 };
  else level = { tier: "hi", pieces: 2 };

  const pinned = i.override?.tier !== undefined || i.override?.pieces !== undefined;
  if (pinned) level = { ...level, ...i.override };
  return { level, adaptive: i.pref === "auto" && !pinned && !i.scripted };
}

/**
 * The next level down, or null at the bottom. hi/3 sheds a piece first; below that the lite LOD
 * (about 3.5x fewer nodes) is worth more than a piece, so hi/2 goes to lite/2; then lite sheds pieces.
 */
export function stepDown({ tier, pieces }: QualityLevel): QualityLevel | null {
  if (tier === "hi") return pieces === 3 ? { tier, pieces: 2 } : { tier: "lite", pieces };
  return pieces > 1 ? { tier, pieces: (pieces - 1) as Pieces } : null;
}

export type BudgetVerdict = "sampling" | "ok" | "slow" | "done";

export interface BudgetOptions {
  /** Frames skipped after start-up or a rebuild, while pipelines compile. */
  warmup?: number;
  /** Frames averaged for one verdict. */
  samples?: number;
  /** Mean frame time above which the level steps down, ms. */
  budgetMs?: number;
}

/** A round whose mean sits within this of its shortest frame is pinned to the display's refresh, not slow. */
const PINNED_SLACK_MS = 2;
/** A gap this long is the tab being hidden or the device asleep, not a slow frame. */
const GAP_MS = 1000;

/**
 * Judges frame times fed one per frame. Returns "sampling" until `samples` frames have been seen
 * after the warm-up, then "slow" or "ok" once. After "slow" the caller rebuilds and calls
 * `reset()` for a fresh round; after "ok" every further frame returns "done".
 */
export class FrameBudget {
  private readonly warmup: number;
  private readonly samples: number;
  private readonly budgetMs: number;
  private seen = 0;
  private count = 0;
  private sum = 0;
  private settled = false;
  private refreshMs = Infinity;

  constructor({ warmup = 30, samples = 90, budgetMs = 20 }: BudgetOptions = {}) {
    this.warmup = warmup;
    this.samples = samples;
    this.budgetMs = budgetMs;
  }

  /** Mean of the round so far, ms; 0 before the first counted frame. */
  get meanMs(): number {
    return this.count ? this.sum / this.count : 0;
  }

  /**
   * `ms` is the wall-clock frame interval, which is only evidence of a slow renderer when it is
   * not simply the display's refresh period: a 30 Hz screen delivers 33 ms frames with the GPU
   * idle. The shortest interval of the round stands in for the refresh period, and the round is
   * slow only when its mean is clearly past that too (missed vsyncs), so a fast GPU on a slow
   * display never steps down.
   */
  push(ms: number): BudgetVerdict {
    if (this.settled) return "done";
    if (this.seen++ < this.warmup || ms > GAP_MS) return "sampling";
    if (ms < this.refreshMs) this.refreshMs = ms;
    this.sum += ms;
    if (++this.count < this.samples) return "sampling";
    if (this.meanMs > this.budgetMs && this.meanMs > this.refreshMs + PINNED_SLACK_MS) return "slow";
    this.settled = true;
    return "ok";
  }

  reset(): void {
    this.seen = this.count = this.sum = 0;
    this.settled = false;
    this.refreshMs = Infinity;
  }
}

/** Highest device pixel ratio worth rendering at: fragment cost grows with its square. */
export function maxPixelRatio(level: QualityLevel, coarsePointer: boolean): number {
  return level.tier === "lite" || coarsePointer ? 1.5 : 2;
}
