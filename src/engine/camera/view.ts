/**
 * What the HUD needs to know about the camera each frame: how many CSS pixels a micrometre is at
 * the distance of the thing being looked at (for the scale bar), which way the pia is on screen
 * (for the compass), and where the selected soma is on screen (for the selection ring).
 *
 * The engine writes into one mutable `HudFrame` per frame; the HUD polls it. Nothing here
 * allocates per frame.
 */

import { Vector3, type PerspectiveCamera } from "three/webgpu";
import { WORLD_UP } from "./fly";
import type { RideState } from "./ride";

export interface HudFrame {
  /** Length the scale bar stands for, µm, and its width in CSS px. */
  scaleUm: number;
  scalePx: number;
  /** Screen angle of the direction toward the pia: 0 is straight up, positive is clockwise, radians. */
  piaAngle: number;
  /** 1 when the pia direction lies in the screen plane, 0 when it points along the view axis. */
  piaLength: number;
  /** Selected soma in CSS px from the canvas's top left; null when nothing is selected or it is behind the camera. */
  selectedScreen: [number, number] | null;
  /** Distance from the camera to the selected soma, µm; 0 when none. */
  selectedDistanceUm: number;
  ride: RideState;
  rideNeuron: number;
  /** Pointer lock is held (mouse look): the HUD shows a crosshair. */
  locked: boolean;
}

export function createHudFrame(): HudFrame {
  return {
    scaleUm: 100,
    scalePx: 100,
    piaAngle: 0,
    piaLength: 1,
    selectedScreen: null,
    selectedDistanceUm: 0,
    ride: "idle",
    rideNeuron: -1,
    locked: false,
  };
}

const NICE_UM = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000];
const MIN_BAR_PX = 40;
const MAX_BAR_PX = 240;

/**
 * The scale bar for a given magnification: 100 µm whenever it fits between 40 and 240 px, otherwise
 * the nearest round length that does, so the label is always a real distance at the bar's width.
 */
export function niceScale(pxPerUm: number): { um: number; px: number } {
  if (!(pxPerUm > 0)) return { um: 100, px: 0 };
  const hundred = 100 * pxPerUm;
  if (hundred >= MIN_BAR_PX && hundred <= MAX_BAR_PX) return { um: 100, px: hundred };
  let um = NICE_UM[0];
  for (const candidate of NICE_UM) {
    if (candidate * pxPerUm <= MAX_BAR_PX) um = candidate;
  }
  return { um, px: um * pxPerUm };
}

/** CSS pixels per µm at view depth `depthUm`, for a vertical FOV and a canvas height in CSS px. */
export function pxPerUmAt(depthUm: number, camera: PerspectiveCamera, heightPx: number): number {
  return (camera.projectionMatrix.elements[5] * 0.5 * heightPx) / Math.max(1, depthUm);
}

const fwd = new Vector3();
const rel = new Vector3();
const local = new Vector3();

/**
 * @param focus What the scale bar is measured at (the selected soma, else the nearest one).
 * @param selected The selected soma, if any.
 */
export function updateHudFrame(
  frame: HudFrame,
  camera: PerspectiveCamera,
  widthPx: number,
  heightPx: number,
  focus: Vector3 | null,
  selected: Vector3 | null,
): void {
  camera.updateMatrixWorld();
  camera.getWorldDirection(fwd);

  const depth = focus ? rel.copy(focus).sub(camera.position).dot(fwd) : 200;
  const scale = niceScale(pxPerUmAt(depth, camera, heightPx));
  frame.scaleUm = scale.um;
  frame.scalePx = scale.px;

  // The pia is at low y, so it lies along WORLD_UP. Rotate that into camera space to read the screen angle.
  local.copy(WORLD_UP).transformDirection(camera.matrixWorldInverse);
  frame.piaLength = Math.min(1, Math.hypot(local.x, local.y));
  frame.piaAngle = Math.atan2(local.x, local.y);

  if (selected) {
    rel.copy(selected).sub(camera.position);
    const d = rel.dot(fwd);
    frame.selectedDistanceUm = rel.length();
    if (d > camera.near) {
      local.copy(selected).applyMatrix4(camera.matrixWorldInverse).applyMatrix4(camera.projectionMatrix);
      frame.selectedScreen = [((local.x + 1) / 2) * widthPx, ((1 - local.y) / 2) * heightPx];
    } else {
      frame.selectedScreen = null;
    }
  } else {
    frame.selectedScreen = null;
    frame.selectedDistanceUm = 0;
  }
}
