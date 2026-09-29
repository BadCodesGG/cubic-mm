/**
 * Cascade tracker: what a stimulated cell set off, read from the bus. Pure, no three.js.
 *
 * `stimulate {neuron}` starts a new cascade rooted there and drops the old one. Then:
 *   reached  post cells the root's pulses have arrived at (`arrive` with pre === root). The sim always
 *            reports the watched neuron's arrivals, so this set is exact.
 *   hop 1    reached cells that spiked within HOP_WINDOW_S after an arrival from the root.
 *   hop 2    cells that spiked within HOP_WINDOW_S after an arrival from a hop-1 cell.
 * When a spike carries its drive generation (the sim sets it: see `sim/model.ts`), "caused" is exact
 * rather than timed: a hop-1 spike is generation 1, and hop 2 counts every later generation, fired
 * by a hop-1 or hop-2 cell. A spike that only happened to follow a landing (generation 255) is not
 * counted, and the landing still names the edge's source. The sim reports every arrival from a
 * driven spike, so hop 2 is then exact too. Spikes without a generation fall back to the timing
 * window, where arrivals from cells the sim is not watching are sampled and hop 2 is a lower bound.
 * A step's spikes and arrivals reach the bus in either order, so each is matched against the other's
 * record whichever comes second. Arrivals from inhibitory cells are ignored: they make nothing fire.
 *
 * Times are simulation seconds. `clock` supplies "now" on that clock: the stimulus time, the sliding
 * timeline window, and staleness (STALE_S without a new caused event) all read it. Spikes are logged
 * per neuron for the life of one cascade; a cascade goes stale within seconds, so the log stays small.
 */

import type { EventBus, SpikeEvent } from "./events";

/** A spike this soon after a pulse landed is counted as caused by it. */
export const HOP_WINDOW_S = 0.05;
/** The drive generation of a spike no stimulus caused. */
const NO_GENERATION = 255;
/** A cascade with no new caused event for this long is no longer active. */
export const STALE_S = 12;
/** Width of the timeline the HUD draws. */
export const TIMELINE_S = 8;
export const MAX_EDGES = 400;

/** `[fromNeuron, toNeuron, time the target fired]`. Hop 1 edges start at the root, hop 2 edges do not. */
export type CascadeEdge = [from: number, to: number, time: number];

export interface CascadeSummary {
  /** The stimulated neuron, or -1 when nothing has been stimulated. */
  root: number;
  reached: number;
  hop1: number;
  /** Every hop after the first. A lower bound when spikes carry no drive generation. */
  hop2: number;
  total: number;
  /** Seconds from the stimulus to the last spike it caused. */
  durationS: number;
  /** Seconds from the stimulus to the last hop-1 spike. */
  hop1SpanS: number;
  /** The hop-1 cells, sorted. */
  hop1Cells: number[];
}

export interface CascadeTick {
  /** Seconds after the stimulus. */
  t: number;
  hop: 1 | 2;
  neuron: number;
}

export interface CascadeOptions {
  clock: () => number;
  /** 1 for inhibitory neurons; their arrivals are not counted as drive. */
  inhibitory?: ArrayLike<number>;
}

/** A spike event, with the drive generation the simulation attaches when it has one. */
export type DrivenSpikeEvent = SpikeEvent & { generation?: number };

interface Landing {
  from: number;
  /** 1 when `from` is a hop-1 cell, 2 when it is a later one. */
  fromHop: 1 | 2;
  time: number;
}

interface LoggedSpike {
  time: number;
  generation?: number;
}

const within = (spike: number, arrival: number) => spike - arrival >= -1e-9 && spike - arrival <= HOP_WINDOW_S + 1e-9;

/** Whether `spike` was caused by a pulse that landed at `arrival`, from the root (1) or from a later cell. */
function causedBy(spike: LoggedSpike, arrival: number, landing: 1 | Landing): boolean {
  if (!within(spike.time, arrival)) return false;
  const g = spike.generation;
  if (landing === 1) return g === undefined || g === 1;
  // Without a generation only hop-1 cells are followed, as the timing heuristic always did.
  if (g === undefined) return landing.fromHop === 1;
  return g >= 2 && g < NO_GENERATION;
}

export class CascadeTracker {
  private root = -1;
  private t0 = 0;
  private lastEvent = 0;
  private lastCaused = -1;
  private lastHop1 = -1;
  private latestNeuron = -1;
  private readonly reached = new Set<number>();
  private readonly rootLandings = new Map<number, number[]>();
  private readonly laterLandings = new Map<number, Landing[]>();
  private readonly hop1 = new Set<number>();
  private readonly hop2 = new Set<number>();
  private hop1Edges: CascadeEdge[] = [];
  private hop2Edges: CascadeEdge[] = [];
  private readonly edgeKeys = new Set<string>();
  private ticks: (CascadeTick & { time: number })[] = [];
  private readonly spikes = new Map<number, LoggedSpike[]>();
  private edgeCache: readonly CascadeEdge[] = [];
  private edgeVersion = 0;
  private cachedVersion = -1;
  private readonly off: (() => void)[];

  constructor(
    bus: EventBus,
    private readonly opts: CascadeOptions,
  ) {
    this.off = [
      bus.on("stimulate", ({ neuron }) => this.start(neuron)),
      bus.on("spike", (e: DrivenSpikeEvent) => this.onSpike(e.neuron, { time: e.time, generation: e.generation })),
      bus.on("arrive", ({ pre, post, time }) => this.onArrive(pre, post, time)),
    ];
  }

  dispose(): void {
    for (const off of this.off) off();
  }

  /** True while a cascade has a root and has seen a caused event within STALE_S. */
  get active(): boolean {
    return this.root >= 0 && this.opts.clock() - this.lastEvent <= STALE_S;
  }

  get rootNeuron(): number {
    return this.root;
  }

  /** Seconds since the stimulus, on the simulation clock. */
  elapsedS(): number {
    return this.root < 0 ? 0 : Math.max(0, this.opts.clock() - this.t0);
  }

  /** The cell that fired most recently as part of the cascade, or -1. */
  latest(): number {
    return this.latestNeuron;
  }

  summary(): CascadeSummary {
    const hop2 = [...this.hop2].filter((n) => !this.hop1.has(n)).length;
    return {
      root: this.root,
      reached: this.reached.size,
      hop1: this.hop1.size,
      hop2,
      total: this.hop1.size + hop2,
      durationS: this.lastCaused >= 0 ? this.lastCaused - this.t0 : 0,
      hop1SpanS: this.lastHop1 >= 0 ? this.lastHop1 - this.t0 : 0,
      hop1Cells: [...this.hop1].sort((a, b) => a - b),
    };
  }

  /** Hop 1 edges then hop 2 edges, at most MAX_EDGES. The same array is returned until an edge is added. */
  edges(): readonly CascadeEdge[] {
    if (this.cachedVersion !== this.edgeVersion) {
      this.cachedVersion = this.edgeVersion;
      const hop2 = this.hop2Edges.filter((e) => !this.hop1.has(e[1]));
      this.edgeCache = [...this.hop1Edges, ...hop2].slice(0, MAX_EDGES);
    }
    return this.edgeCache;
  }

  /** Spikes the cascade caused in the last TIMELINE_S, oldest first. */
  timeline(): CascadeTick[] {
    const from = this.opts.clock() - TIMELINE_S;
    return this.ticks.filter((k) => k.time >= from).map(({ t, hop, neuron }) => ({ t, hop, neuron }));
  }

  private start(neuron: number): void {
    this.root = neuron;
    this.t0 = this.opts.clock();
    this.lastEvent = this.t0;
    this.lastCaused = -1;
    this.lastHop1 = -1;
    this.latestNeuron = -1;
    this.reached.clear();
    this.rootLandings.clear();
    this.laterLandings.clear();
    this.hop1.clear();
    this.hop2.clear();
    this.hop1Edges = [];
    this.hop2Edges = [];
    this.edgeKeys.clear();
    this.ticks = [];
    this.spikes.clear();
    this.edgeVersion++;
  }

  private caused(hop: 1 | 2, neuron: number, time: number): void {
    this.ticks.push({ t: time - this.t0, hop, neuron, time });
    this.lastEvent = Math.max(this.lastEvent, time);
    if (time >= this.lastCaused) {
      this.lastCaused = time;
      this.latestNeuron = neuron;
    }
    if (hop === 1) this.lastHop1 = Math.max(this.lastHop1, time);
  }

  private addEdge(hop: 1 | 2, from: number, to: number, time: number): void {
    const key = `${from}>${to}`;
    const list = hop === 1 ? this.hop1Edges : this.hop2Edges;
    if (this.edgeKeys.has(key) || list.length >= MAX_EDGES) return;
    this.edgeKeys.add(key);
    list.push([from, to, time]);
    this.edgeVersion++;
  }

  private landHop1(neuron: number, time: number): void {
    if (!this.hop1.has(neuron)) {
      this.hop1.add(neuron);
      this.caused(1, neuron, time);
    }
    this.addEdge(1, this.root, neuron, time);
  }

  private landHop2(from: number, neuron: number, time: number): void {
    if (this.hop1.has(neuron)) return;
    if (!this.hop2.has(neuron)) {
      this.hop2.add(neuron);
      this.caused(2, neuron, time);
    }
    this.addEdge(2, from, neuron, time);
  }

  private onSpike(neuron: number, spike: LoggedSpike): void {
    if (this.root < 0 || !this.active) return;
    const log = this.spikes.get(neuron);
    if (log) log.push(spike);
    else this.spikes.set(neuron, [spike]);
    if (neuron === this.root || spike.time < this.t0) return;

    if (this.rootLandings.get(neuron)?.some((ta) => causedBy(spike, ta, 1))) {
      this.landHop1(neuron, spike.time);
      return;
    }
    for (const l of this.laterLandings.get(neuron) ?? []) {
      if (causedBy(spike, l.time, l)) this.landHop2(l.from, neuron, spike.time);
    }
  }

  private onArrive(pre: number, post: number, time: number): void {
    if (this.root < 0 || !this.active || time < this.t0) return;
    if (this.opts.inhibitory?.[pre]) return;
    const fired = (landing: 1 | Landing) =>
      post === this.root ? undefined : this.spikes.get(post)?.find((s) => causedBy(s, time, landing))?.time;
    if (pre === this.root) {
      this.reached.add(post);
      this.lastEvent = Math.max(this.lastEvent, time);
      const list = this.rootLandings.get(post);
      if (list) list.push(time);
      else this.rootLandings.set(post, [time]);
      const at = fired(1);
      if (at !== undefined) this.landHop1(post, at);
    } else if (this.hop1.has(pre) || this.hop2.has(pre)) {
      const landing: Landing = { from: pre, fromHop: this.hop1.has(pre) ? 1 : 2, time };
      const list = this.laterLandings.get(post);
      if (list) list.push(landing);
      else this.laterLandings.set(post, [landing]);
      const at = fired(landing);
      if (at !== undefined) this.landHop2(pre, post, at);
    }
  }
}
