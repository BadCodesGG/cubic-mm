/**
 * Camera motion. Two modes:
 *  - `FlyControls`: pointer-lock fly camera, WASD to move, Q/E down/up, shift to boost, mouse to
 *    look, with inertia on both movement and look.
 *  - `heroPath(t)`: a slow scripted drift near one neuron, so screenshots are reproducible.
 *
 * World units are µm and y is cortical depth with the pia at low y, so "up" on screen is -y.
 */

import { Vector3, type PerspectiveCamera } from "three/webgpu";

export const WORLD_UP = new Vector3(0, -1, 0);

const LOOK_SENSITIVITY = 0.0022;
const BASE_SPEED = 45; // µm per second
const BOOST = 4;
const MAX_PITCH = Math.PI / 2 - 0.05;

export class FlyControls {
  enabled = true;
  private yaw = 0;
  private pitch = 0;
  private targetYaw = 0;
  private targetPitch = 0;
  private readonly velocity = new Vector3();
  private readonly keys = new Set<string>();
  private readonly forward = new Vector3();
  private readonly right = new Vector3();
  private readonly wish = new Vector3();
  private readonly look = new Vector3();

  constructor(
    private readonly camera: PerspectiveCamera,
    private readonly dom: HTMLElement,
  ) {
    camera.up.copy(WORLD_UP);
    dom.addEventListener("click", this.onClick);
    document.addEventListener("pointerlockerror", this.onLockError);
    document.addEventListener("mousemove", this.onMouseMove);
    window.addEventListener("keydown", this.onKeyDown);
    window.addEventListener("keyup", this.onKeyUp);
    window.addEventListener("blur", this.onBlur);
  }

  /** Point the camera at `target` and make that the look direction the controls start from. */
  lookAt(target: Vector3): void {
    const dir = target.clone().sub(this.camera.position).normalize();
    this.pitch = this.targetPitch = Math.asin(Math.max(-1, Math.min(1, -dir.y)));
    this.yaw = this.targetYaw = Math.atan2(dir.x, dir.z);
    this.apply();
  }

  update(dt: number): void {
    if (!this.enabled) return;
    const look = 1 - Math.exp(-dt * 14);
    this.yaw += (this.targetYaw - this.yaw) * look;
    this.pitch += (this.targetPitch - this.pitch) * look;
    this.apply();

    this.right.crossVectors(this.forward, WORLD_UP).normalize();
    this.wish.set(0, 0, 0);
    if (this.keys.has("KeyW")) this.wish.add(this.forward);
    if (this.keys.has("KeyS")) this.wish.sub(this.forward);
    if (this.keys.has("KeyD")) this.wish.add(this.right);
    if (this.keys.has("KeyA")) this.wish.sub(this.right);
    if (this.keys.has("KeyE")) this.wish.add(WORLD_UP);
    if (this.keys.has("KeyQ")) this.wish.sub(WORLD_UP);
    if (this.wish.lengthSq() > 0) this.wish.normalize();
    const boosted = this.keys.has("ShiftLeft") || this.keys.has("ShiftRight");
    this.wish.multiplyScalar(BASE_SPEED * (boosted ? BOOST : 1));
    this.velocity.lerp(this.wish, 1 - Math.exp(-dt * 5));
    this.camera.position.addScaledVector(this.velocity, dt);
  }

  dispose(): void {
    this.dom.removeEventListener("click", this.onClick);
    document.removeEventListener("pointerlockerror", this.onLockError);
    document.removeEventListener("mousemove", this.onMouseMove);
    window.removeEventListener("keydown", this.onKeyDown);
    window.removeEventListener("keyup", this.onKeyUp);
    window.removeEventListener("blur", this.onBlur);
    if (document.pointerLockElement === this.dom) document.exitPointerLock();
  }

  private apply(): void {
    const cp = Math.cos(this.pitch);
    this.forward.set(cp * Math.sin(this.yaw), -Math.sin(this.pitch), cp * Math.cos(this.yaw));
    this.look.copy(this.camera.position).add(this.forward);
    this.camera.lookAt(this.look);
  }

  private onClick = (e: MouseEvent) => {
    // Pointer lock is refused inside embedded frames and by some browsers; a refusal is not an error
    // for us, the visitor just keeps the unlocked camera. Chrome reports it both as a rejected promise
    // and as a `pointerlockerror` event, so both are swallowed.
    if (!e.isTrusted || !this.enabled || document.pointerLockElement === this.dom) return;
    try {
      const result = this.dom.requestPointerLock() as Promise<void> | undefined;
      result?.catch(() => {});
    } catch {
      // Older engines throw synchronously instead of rejecting.
    }
  };

  private onLockError = () => {};

  private onMouseMove = (e: MouseEvent) => {
    if (!this.enabled || document.pointerLockElement !== this.dom) return;
    this.targetYaw -= e.movementX * LOOK_SENSITIVITY;
    this.targetPitch = Math.max(-MAX_PITCH, Math.min(MAX_PITCH, this.targetPitch - e.movementY * LOOK_SENSITIVITY));
  };

  private onKeyDown = (e: KeyboardEvent) => {
    this.keys.add(e.code);
  };

  private onKeyUp = (e: KeyboardEvent) => {
    this.keys.delete(e.code);
  };

  private onBlur = () => {
    this.keys.clear();
  };
}

export interface HeroAnchor {
  /** Soma centre, µm. */
  soma: Vector3;
  /** Unit direction the axon heads off in, flattened toward the horizontal. */
  axis: Vector3;
}

export interface CameraPose {
  position: Vector3;
  target: Vector3;
}

/**
 * The scripted hero drift: start about 150 µm behind and beside the soma, look out along its
 * axon, and dolly slowly forward with a faint lateral sway. Pure in `t`, so a given time always
 * gives the same frame.
 */
export function heroPath(t: number, anchor: HeroAnchor, out: CameraPose): CameraPose {
  const { soma, axis } = anchor;
  const side = new Vector3().crossVectors(axis, WORLD_UP).normalize();
  const up = new Vector3().crossVectors(side, axis).normalize();
  const dolly = 3.5 * t;
  const sway = Math.sin(t * 0.21) * 6;
  out.position
    .copy(soma)
    .addScaledVector(axis, -105 + dolly)
    .addScaledVector(side, 95 + sway)
    .addScaledVector(up, 38);
  out.target
    .copy(soma)
    .addScaledVector(axis, 190 + dolly * 0.5)
    .addScaledVector(side, 10)
    .addScaledVector(up, -10);
  return out;
}
