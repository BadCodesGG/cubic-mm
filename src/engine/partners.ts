/**
 * The connectome as a graph: who each neuron talks to, and who talks to it. Pure, no three.js.
 *
 * The synapse table lists every synapse once, grouped by presynaptic neuron (`preOffsets`). The
 * index folds it into one edge per (pre, post) pair with the synapse count and summed size, and
 * lays those edges out twice with offsets, like `preOffsets`: by presynaptic neuron for `outputs`,
 * and by postsynaptic neuron (a counting sort over the edges) for `inputs`. Building it is a
 * single pass over the synapses plus one over the edges, a few milliseconds for the whole volume.
 *
 * `WiringState` is what the scene draws from: the hovered cell, the cell the visitor pinned, and a
 * timed preview (the search reward). It reads the bus and holds no three.js objects.
 */

import type { EventBus } from "./events";
import type { SynapseTable } from "./format";

export interface Partner {
  neuron: number;
  /** Synapses between the two cells, in the direction of this list. */
  synapses: number;
  /** Sum of the synapses' size buckets (0..255 each): a weight hint. */
  size: number;
}

export interface PartnerStats {
  /** Distinct cells this one makes synapses onto. */
  outputs: number;
  /** Distinct cells that make synapses onto this one. */
  inputs: number;
  outputSynapses: number;
  inputSynapses: number;
  /** How many of the input cells are inhibitory. */
  inhibitoryInputs: number;
}

const bySynapses = (a: Partner, b: Partner) => b.synapses - a.synapses || b.size - a.size || a.neuron - b.neuron;

export class PartnerIndex {
  /** Distinct (pre, post) pairs. */
  readonly edgeCount: number;
  /** The most partners (outputs plus inputs) any one cell has. */
  readonly maxPartners: number;
  private readonly outOffsets: Uint32Array;
  private readonly outNeuron: Uint16Array;
  private readonly outCount: Uint32Array;
  private readonly outSize: Uint32Array;
  private readonly inOffsets: Uint32Array;
  private readonly inNeuron: Uint16Array;
  /** Index into the out arrays of the edge each input entry mirrors. */
  private readonly inEdge: Uint32Array;

  constructor(
    table: SynapseTable,
    private readonly inhibitory: ArrayLike<number>,
  ) {
    const n = table.neuronCount;
    const { preOffsets, post, size } = table;

    // Outgoing: one pass over the synapses. `slot[b]` is the edge (a -> b) of the neuron being read;
    // a slot below `first` belongs to an earlier neuron (or is unset) and is stale.
    const outOffsets = new Uint32Array(n + 1);
    const neuron = new Uint16Array(table.count);
    const count = new Uint32Array(table.count);
    const sum = new Uint32Array(table.count);
    const slot = new Int32Array(n).fill(-1);
    let e = 0;
    for (let a = 0; a < n; a++) {
      outOffsets[a] = e;
      const first = e;
      for (let s = preOffsets[a]; s < preOffsets[a + 1]; s++) {
        const b = post[s];
        let k = slot[b];
        if (k < first) {
          k = e++;
          slot[b] = k;
          neuron[k] = b;
        }
        count[k]++;
        sum[k] += size[s];
      }
    }
    outOffsets[n] = e;
    this.edgeCount = e;
    this.outOffsets = outOffsets;
    this.outNeuron = neuron.slice(0, e);
    this.outCount = count.slice(0, e);
    this.outSize = sum.slice(0, e);

    // Incoming: counting sort of the edges by postsynaptic neuron.
    const inOffsets = new Uint32Array(n + 1);
    for (let k = 0; k < e; k++) inOffsets[this.outNeuron[k] + 1]++;
    for (let b = 0; b < n; b++) inOffsets[b + 1] += inOffsets[b];
    const cursor = inOffsets.slice(0, n);
    const inNeuron = new Uint16Array(e);
    const inEdge = new Uint32Array(e);
    for (let a = 0; a < n; a++) {
      for (let k = outOffsets[a]; k < outOffsets[a + 1]; k++) {
        const at = cursor[this.outNeuron[k]]++;
        inNeuron[at] = a;
        inEdge[at] = k;
      }
    }
    this.inOffsets = inOffsets;
    this.inNeuron = inNeuron;
    this.inEdge = inEdge;

    let max = 0;
    for (let a = 0; a < n; a++) max = Math.max(max, outOffsets[a + 1] - outOffsets[a] + inOffsets[a + 1] - inOffsets[a]);
    this.maxPartners = max;
  }

  get neuronCount(): number {
    return this.outOffsets.length - 1;
  }

  /** Cells this one synapses onto, and cells that synapse onto it, each most synapses first. */
  partnersOf(neuron: number): { outputs: Partner[]; inputs: Partner[] } {
    if (!(neuron >= 0 && neuron < this.neuronCount)) return { outputs: [], inputs: [] };
    const outputs: Partner[] = [];
    for (let k = this.outOffsets[neuron]; k < this.outOffsets[neuron + 1]; k++) {
      outputs.push({ neuron: this.outNeuron[k], synapses: this.outCount[k], size: this.outSize[k] });
    }
    const inputs: Partner[] = [];
    for (let k = this.inOffsets[neuron]; k < this.inOffsets[neuron + 1]; k++) {
      const edge = this.inEdge[k];
      inputs.push({ neuron: this.inNeuron[k], synapses: this.outCount[edge], size: this.outSize[edge] });
    }
    return { outputs: outputs.sort(bySynapses), inputs: inputs.sort(bySynapses) };
  }

  partnerStats(neuron: number): PartnerStats {
    const stats: PartnerStats = { outputs: 0, inputs: 0, outputSynapses: 0, inputSynapses: 0, inhibitoryInputs: 0 };
    if (!(neuron >= 0 && neuron < this.neuronCount)) return stats;
    stats.outputs = this.outOffsets[neuron + 1] - this.outOffsets[neuron];
    stats.inputs = this.inOffsets[neuron + 1] - this.inOffsets[neuron];
    for (let k = this.outOffsets[neuron]; k < this.outOffsets[neuron + 1]; k++) stats.outputSynapses += this.outCount[k];
    for (let k = this.inOffsets[neuron]; k < this.inOffsets[neuron + 1]; k++) {
      stats.inputSynapses += this.outCount[this.inEdge[k]];
      if (this.inhibitory[this.inNeuron[k]]) stats.inhibitoryInputs++;
    }
    return stats;
  }
}

export function buildPartnerIndex(table: SynapseTable, inhibitory: ArrayLike<number>): PartnerIndex {
  return new PartnerIndex(table, inhibitory);
}

/* ------------------------------------------------------------------ the lines of one graph */

/** Most lines one cell's graph draws. */
export const MAX_LINES = 400;

export const enum LineKind {
  Output = 0,
  Input = 1,
  /** An input from an inhibitory cell. */
  InhibitoryInput = 2,
}

export interface WiringLine {
  neuron: number;
  kind: LineKind;
  synapses: number;
}

/** The graph's lines for `neuron`: outputs and inputs together, most synapses first, at most `cap`. A cell wired both ways gets two. */
export function wiringLines(index: PartnerIndex, inhibitory: ArrayLike<number>, neuron: number, cap = MAX_LINES): WiringLine[] {
  const { outputs, inputs } = index.partnersOf(neuron);
  const lines: WiringLine[] = [];
  for (const p of outputs) if (p.neuron !== neuron) lines.push({ neuron: p.neuron, kind: LineKind.Output, synapses: p.synapses });
  for (const p of inputs) {
    if (p.neuron === neuron) continue;
    lines.push({ neuron: p.neuron, kind: inhibitory[p.neuron] ? LineKind.InhibitoryInput : LineKind.Input, synapses: p.synapses });
  }
  lines.sort((a, b) => b.synapses - a.synapses || a.neuron - b.neuron || a.kind - b.kind);
  return lines.length > cap ? lines.slice(0, cap) : lines;
}

/** Brightness 0.3..1 of a line by its synapse count, on a log scale that reaches 1 at 16 synapses. */
export function lineWeight(synapses: number): number {
  return Math.min(1, 0.3 + 0.7 * (Math.log(Math.max(1, synapses)) / Math.log(16)));
}

/* ------------------------------------------------------------------ fading */

export const FADE_IN_S = 0.15;
export const FADE_OUT_S = 0.25;
/** Level of the pinned graph while another cell is hovered. */
export const PIN_DIM = 0.55;

/** Moves `level` toward `target`: up over FADE_IN_S, down over FADE_OUT_S (both for a full 0..1 swing). */
export function approach(level: number, target: number, dtS: number): number {
  return level < target ? Math.min(target, level + dtS / FADE_IN_S) : Math.max(target, level - dtS / FADE_OUT_S);
}

/* ------------------------------------------------------------------ what is shown */

/** How long a search or chip flight shows the graph on arrival. */
export const PREVIEW_S = 4;

/**
 * Which cells' graphs are wanted: the hovered one and the pinned one. Reads `hover`, `partners`,
 * `select` and `tour`, and `tick(dt)` runs the preview timer.
 *
 * A pin belongs to a cell: selecting another cell (or none) drops it, so Esc and Deselect clear it.
 * A preview is a pin that releases itself; the panel does not show it as pinned, so pressing the
 * toggle during one makes it a real pin.
 */
export class WiringState {
  private hovered = -1;
  private pin = -1;
  private previewed = -1;
  private previewLeft = 0;
  private selected = -1;
  private touring = false;
  private readonly offs: (() => void)[];

  constructor(
    bus: EventBus,
    private readonly neuronCount: number,
  ) {
    const valid = (n: number) => Number.isInteger(n) && n >= 0 && n < neuronCount;
    this.offs = [
      bus.on("hover", ({ neuron }) => (this.hovered = valid(neuron) ? neuron : -1)),
      bus.on("partners", ({ neuron, show }) => {
        if (!valid(neuron)) return;
        if (show) {
          this.pin = neuron;
          if (this.previewed === neuron) this.previewed = -1;
        } else {
          if (this.pin === neuron) this.pin = -1;
          if (this.previewed === neuron) this.previewed = -1;
        }
      }),
      bus.on("select", ({ neuron }) => {
        this.selected = neuron;
        if (this.pin !== neuron) this.pin = -1;
        if (this.previewed !== neuron) this.previewed = -1;
      }),
      bus.on("tour", ({ running }) => {
        this.touring = running;
        if (running) this.pin = this.previewed = -1;
      }),
    ];
  }

  /** The cell the visitor pinned, or -1. */
  get pinned(): number {
    return this.pin;
  }

  /** Show `neuron`'s graph for `seconds`, then let it go. */
  preview(neuron: number, seconds = PREVIEW_S): void {
    if (this.touring || !(neuron >= 0 && neuron < this.neuronCount)) return;
    this.previewed = neuron;
    this.previewLeft = seconds;
  }

  /** The cells to draw: the pointer's (never the pinned one twice) and the pinned or previewed one. */
  shown(): { hover: number; pin: number } {
    const pin = this.pin >= 0 ? this.pin : this.previewed;
    const hover = this.touring || this.hovered === pin ? -1 : this.hovered;
    return { hover, pin };
  }

  tick(dtS: number): void {
    if (this.previewed < 0) return;
    this.previewLeft -= dtS;
    if (this.previewLeft <= 0) this.previewed = -1;
  }

  /** `G` pins the selection's graph, or unpins it. Returns whether the key was used. */
  onKey(e: { code: string; repeat: boolean; ctrlKey: boolean; metaKey: boolean; altKey: boolean; target: unknown }, bus: EventBus): boolean {
    if (e.code !== "KeyG" || e.repeat || e.ctrlKey || e.metaKey || e.altKey || this.selected < 0) return false;
    if (e.target instanceof Element && e.target.closest("input, textarea, select, [contenteditable]")) return false;
    bus.emit("partners", { neuron: this.selected, show: this.pin !== this.selected });
    return true;
  }

  dispose(): void {
    for (const off of this.offs) off();
  }
}
