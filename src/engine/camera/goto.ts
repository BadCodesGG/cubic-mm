/**
 * Scripted flights: an eased move of the camera from where it is to a pose, then a hand-back to
 * `FlyControls`. Shared by "return to the cluster", "jump to this cell", the shareable link
 * restore and the intro tour, so every automatic camera move feels the same.
 *
 * The flight owns the camera while `active`; the app loop calls `update(dt)` and skips the
 * controls while it returns true, exactly as it does for the ride camera.
 */

import { Vector3, type PerspectiveCamera } from "three/webgpu";
import { WORLD_UP, type FlyControls } from "./fly";

export interface Pose {
  position: Vector3;
  target: Vector3;
}

/** Smoothstep with zero velocity at both ends. */
export function ease(k: number): number {
  const t = Math.max(0, Math.min(1, k));
  return t * t * (3 - 2 * t);
}

/** A viewing pose for a soma: back along `away` (normalised) by `distanceUm`, looking at the soma. */
export function poseAround(soma: Vector3, away: Vector3, distanceUm: number, out: Pose): Pose {
  out.position.copy(soma).addScaledVector(away, distanceUm).addScaledVector(WORLD_UP, distanceUm * 0.25);
  out.target.copy(soma);
  return out;
}

export class Flight {
  private from: Pose = { position: new Vector3(), target: new Vector3() };
  private to: Pose = { position: new Vector3(), target: new Vector3() };
  private elapsed = 0;
  private duration = 0;
  private running = false;
  private onDone: (() => void) | null = null;
  private readonly dir = new Vector3();

  constructor(
    private readonly camera: PerspectiveCamera,
    private readonly controls: FlyControls | null,
  ) {}

  get active(): boolean {
    return this.running;
  }

  /**
   * Fly to `pose` over `seconds`. Duration scales with distance when `seconds` is omitted:
   * 1.2 s plus 1 s per 300 µm, capped at 4 s.
   */
  start(pose: Pose, seconds?: number, onDone?: () => void): void {
    this.from.position.copy(this.camera.position);
    this.camera.getWorldDirection(this.dir);
    this.to.position.copy(pose.position);
    this.to.target.copy(pose.target);
    // Start the look point as deep as the destination's, so the gaze turns evenly rather than
    // swinging while a near point is lerped toward a far one.
    this.from.target.copy(this.camera.position).addScaledVector(this.dir, Math.max(50, this.camera.position.distanceTo(pose.target)));
    const distance = this.from.position.distanceTo(this.to.position);
    this.duration = seconds ?? Math.min(4, 1.2 + distance / 300);
    this.elapsed = 0;
    this.running = true;
    this.onDone = onDone ?? null;
    if (this.controls) this.controls.enabled = false;
  }

  cancel(): void {
    if (!this.running) return;
    this.running = false;
    this.onDone = null;
    this.handBack();
  }

  /** Advances the flight. Returns true while it owns the camera. */
  update(dt: number): boolean {
    if (!this.running) return false;
    this.elapsed += dt;
    const k = ease(this.duration > 0 ? this.elapsed / this.duration : 1);
    this.camera.position.lerpVectors(this.from.position, this.to.position, k);
    this.dir.lerpVectors(this.from.target, this.to.target, k);
    this.camera.up.copy(WORLD_UP);
    this.camera.lookAt(this.dir);
    if (this.elapsed >= this.duration) {
      this.running = false;
      const done = this.onDone;
      this.onDone = null;
      this.handBack();
      done?.();
    }
    return true;
  }

  private handBack(): void {
    if (!this.controls) return;
    this.controls.lookAt(this.to.target);
    this.controls.enabled = true;
  }
}
