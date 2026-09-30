/**
 * Picking and the selection keys.
 *
 * A click selects the neuron whose soma projects nearest the cursor within 24 px, preferring nearer
 * somas; a click that lands on nothing clears the selection. While the mouse is captured
 * (pointer lock) there is no cursor, so the click picks at the screen centre, where the HUD draws
 * a crosshair. A press that travelled more than a few pixels is a look drag, not a click.
 *
 *   click        select the neuron under the cursor (or none)
 *   double-click stimulate it
 *   Space        stimulate the selected neuron
 *   R            ride the selected neuron's next spike
 *   Esc          clear the selection (unless a ride just consumed the key)
 */

import { Vector3, type PerspectiveCamera } from "three/webgpu";
import type { EventBus } from "./events";

export const PICK_RADIUS_PX = 24;
/** A candidate's screen distance is charged this many px per µm of depth, so a nearer soma wins a near tie. */
const DEPTH_PENALTY_PX_PER_UM = 0.04;
const DRAG_PX = 5;

const view = new Vector3();
const ndc = new Vector3();

/**
 * The soma nearest the point (px, py), in CSS pixels from the canvas's top left, within `radiusPx`.
 * Somas behind the near plane are ignored. Returns -1 when nothing is in range.
 * Score is screen distance plus `DEPTH_PENALTY_PX_PER_UM` per µm of depth.
 */
export function pickSoma(
  somaPos: Float32Array,
  camera: PerspectiveCamera,
  widthPx: number,
  heightPx: number,
  px: number,
  py: number,
  radiusPx = PICK_RADIUS_PX,
): number {
  camera.updateMatrixWorld();
  let best = -1;
  let bestScore = Infinity;
  for (let i = 0; i < somaPos.length / 3; i++) {
    view.set(somaPos[i * 3], somaPos[i * 3 + 1], somaPos[i * 3 + 2]).applyMatrix4(camera.matrixWorldInverse);
    const depth = -view.z;
    if (depth <= camera.near) continue;
    ndc.copy(view).applyMatrix4(camera.projectionMatrix);
    const dist = Math.hypot(((ndc.x + 1) / 2) * widthPx - px, ((1 - ndc.y) / 2) * heightPx - py);
    if (dist > radiusPx) continue;
    const score = dist + depth * DEPTH_PENALTY_PX_PER_UM;
    if (score < bestScore) {
      bestScore = score;
      best = i;
    }
  }
  return best;
}

const INTERACTIVE = "button, a, input, select, textarea, summary, [role='button'], [contenteditable]";

export class Picker {
  private selected = -1;
  // --- r4/graph: the soma under the pointer, so `hover` fires when it changes and not on every move ---
  private hovered = -1;
  // --- end r4/graph ---
  private down: { x: number; y: number } | null = null;
  private readonly offs: (() => void)[] = [];

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly camera: PerspectiveCamera,
    private readonly somaPos: Float32Array,
    private readonly bus: EventBus,
  ) {
    this.offs.push(bus.on("select", (e) => (this.selected = e.neuron)));
    canvas.addEventListener("pointerdown", this.onPointerDown);
    canvas.addEventListener("pointermove", this.onPointerMove);
    canvas.addEventListener("pointerleave", this.onPointerLeave); // r4/graph
    canvas.addEventListener("click", this.onClick);
    canvas.addEventListener("dblclick", this.onDoubleClick);
    window.addEventListener("keydown", this.onKeyDown);
  }

  get selection(): number {
    return this.selected;
  }

  /** The neuron under a mouse event, or at the screen centre while the mouse is captured. */
  pickAt(e: MouseEvent): number {
    const rect = this.canvas.getBoundingClientRect();
    const locked = document.pointerLockElement === this.canvas;
    const x = locked ? rect.width / 2 : e.clientX - rect.left;
    const y = locked ? rect.height / 2 : e.clientY - rect.top;
    return pickSoma(this.somaPos, this.camera, rect.width, rect.height, x, y);
  }

  dispose(): void {
    for (const off of this.offs) off();
    this.canvas.removeEventListener("pointerdown", this.onPointerDown);
    this.canvas.removeEventListener("pointermove", this.onPointerMove);
    this.canvas.removeEventListener("pointerleave", this.onPointerLeave); // r4/graph
    this.canvas.removeEventListener("click", this.onClick);
    this.canvas.removeEventListener("dblclick", this.onDoubleClick);
    window.removeEventListener("keydown", this.onKeyDown);
    this.canvas.style.cursor = "";
  }

  private onPointerDown = (e: PointerEvent) => {
    this.down = { x: e.clientX, y: e.clientY };
  };

  /** Hover feedback: a pointer cursor over a pickable soma. */
  private onPointerMove = (e: PointerEvent) => {
    if (e.pointerType !== "mouse") return;
    // --- r4/graph: a drag or a captured mouse is looking around, not pointing at a cell ---
    if (e.buttons !== 0 || document.pointerLockElement === this.canvas) {
      this.setHover(-1);
      return;
    }
    // --- end r4/graph ---
    const hit = this.pickAt(e);
    this.canvas.style.cursor = hit >= 0 ? "pointer" : "";
    this.setHover(hit); // r4/graph
  };

  // --- r4/graph ---
  private onPointerLeave = () => {
    this.canvas.style.cursor = "";
    this.setHover(-1);
  };

  private setHover(neuron: number): void {
    if (neuron === this.hovered) return;
    this.hovered = neuron;
    this.bus.emit("hover", { neuron });
  }
  // --- end r4/graph ---

  private onClick = (e: MouseEvent) => {
    const locked = document.pointerLockElement === this.canvas;
    if (!locked && this.down && Math.hypot(e.clientX - this.down.x, e.clientY - this.down.y) > DRAG_PX) return;
    const hit = this.pickAt(e);
    // A miss while the mouse is captured is just the visitor clicking to look around; keep the selection.
    if (hit < 0 && locked) return;
    if (hit !== this.selected) this.bus.emit("select", { neuron: hit });
  };

  private onDoubleClick = (e: MouseEvent) => {
    const hit = this.pickAt(e);
    if (hit < 0) return;
    if (hit !== this.selected) this.bus.emit("select", { neuron: hit });
    this.bus.emit("stimulate", { neuron: hit });
  };

  private onKeyDown = (e: KeyboardEvent) => {
    if (e.repeat || e.ctrlKey || e.metaKey || e.altKey || this.selected < 0) return;
    const onControl = e.target instanceof Element && e.target.closest(INTERACTIVE) !== null;
    if (e.code === "Space" && !onControl) {
      e.preventDefault();
      this.bus.emit("stimulate", { neuron: this.selected });
    } else if (e.code === "KeyR") {
      this.bus.emit("ride", { neuron: this.selected });
    } else if (e.code === "Escape" && !e.defaultPrevented) {
      this.bus.emit("select", { neuron: -1 });
    }
  };
}
