/**
 * Shareable links: the camera and the selection, kept in the URL hash so the address bar is always
 * a link to what the visitor is looking at.
 *
 *   #c=<x>,<y>,<z>,<yaw>,<pitch>[&n=<neuron>]&v=<count>
 *
 * Position is µm to one decimal, angles are radians to three. `n` is the neuron's index in the
 * dataset, not its root id: it is a few characters instead of eighteen digits, and the index is
 * stable for a given data build. That stability is exactly what `v`, the dataset's neuron count,
 * guards: a link made against a different build is ignored rather than selecting the wrong cell.
 *
 * Pure module: no three, no DOM. `app.ts` supplies the camera and the `history.replaceState` writer.
 */

export interface ViewState {
  x: number;
  y: number;
  z: number;
  /** Radians, 0 faces +z, grows toward +x. Same convention as `FlyControls`. */
  yaw: number;
  /** Radians, positive looks toward -y (up on screen). */
  pitch: number;
  /** Dataset neuron index, -1 for none. */
  neuron: number;
}

const TWO_PI = Math.PI * 2;
const MAX_COORD_UM = 1e6;
const NUMBER = /^-?\d+(\.\d+)?$/;
const INDEX = /^\d+$/;

/** `toFixed` without the "-0.0" it gives a value that rounds to zero. */
function fixed(v: number, digits: number): string {
  const s = v.toFixed(digits);
  return /^-0(\.0+)?$/.test(s) ? s.slice(1) : s;
}

function wrapYaw(yaw: number): number {
  const w = yaw - TWO_PI * Math.round(yaw / TWO_PI);
  return w <= -Math.PI ? w + TWO_PI : w;
}

/** The hash body (no leading `#`) for a view of a dataset with `neuronCount` neurons. */
export function encodeView(view: ViewState, neuronCount: number): string {
  const c = [fixed(view.x, 1), fixed(view.y, 1), fixed(view.z, 1), fixed(wrapYaw(view.yaw), 3), fixed(view.pitch, 3)];
  return `c=${c.join(",")}${view.neuron >= 0 ? `&n=${view.neuron}` : ""}&v=${neuronCount}`;
}

/**
 * The view a hash describes, or null for anything else: garbage, a partial link, values out of
 * range, a neuron index outside the dataset, or a link made against a dataset of a different size.
 */
export function decodeView(hash: string, neuronCount: number): ViewState | null {
  const params = new Map<string, string>();
  for (const part of (hash.startsWith("#") ? hash.slice(1) : hash).split("&")) {
    const eq = part.indexOf("=");
    if (eq <= 0) return null;
    const key = part.slice(0, eq);
    if (params.has(key)) return null;
    params.set(key, part.slice(eq + 1));
  }

  const c = params.get("c")?.split(",");
  if (!c || c.length !== 5 || !c.every((s) => NUMBER.test(s))) return null;
  const [x, y, z, yaw, pitch] = c.map(Number);
  if (Math.abs(x) > MAX_COORD_UM || Math.abs(y) > MAX_COORD_UM || Math.abs(z) > MAX_COORD_UM) return null;
  if (Math.abs(yaw) > 1000 || Math.abs(pitch) > Math.PI / 2) return null;

  const v = params.get("v");
  if (v !== undefined && (!INDEX.test(v) || Number(v) !== neuronCount)) return null;

  let neuron = -1;
  const n = params.get("n");
  if (n !== undefined) {
    if (!INDEX.test(n) || Number(n) >= neuronCount) return null;
    neuron = Number(n);
  }
  return { x, y, z, yaw, pitch, neuron };
}

/** Yaw and pitch of a look direction, in `FlyControls`'s convention. The direction need not be unit length. */
export function lookAngles(dx: number, dy: number, dz: number): { yaw: number; pitch: number } {
  const len = Math.hypot(dx, dy, dz) || 1;
  return { yaw: Math.atan2(dx, dz), pitch: Math.asin(Math.max(-1, Math.min(1, -dy / len))) };
}

/** The unit look direction for a yaw and pitch: the inverse of `lookAngles`. */
export function lookDirection(yaw: number, pitch: number): [number, number, number] {
  const cp = Math.cos(pitch);
  return [cp * Math.sin(yaw), -Math.sin(pitch), cp * Math.cos(yaw)];
}

export interface LinkSyncOptions {
  getView(): ViewState;
  /** Neuron count of the loaded dataset, written as `v`. */
  count: number;
  /** Receives the hash body; the app writes it with `history.replaceState`. */
  write(hash: string): void;
  /** Least time between writes while the camera moves. */
  intervalMs?: number;
}

/**
 * Keeps the URL hash current without spamming it. `update(now)` runs every frame and writes at
 * most once per `intervalMs`, and only when the link changed. The view the page started at is the
 * baseline, so a visitor who has not moved keeps a clean URL (and the intro tour, which is skipped
 * when a link is present). `flush()` writes now, for a selection change or Copy link.
 */
export class LinkSync {
  private last: string | null = null;
  private lastWriteAt = -Infinity;
  private now = 0;
  private paused = false;
  private readonly intervalMs: number;

  constructor(private readonly opts: LinkSyncOptions) {
    this.intervalMs = opts.intervalMs ?? 2000;
  }

  private current(): string {
    return encodeView(this.opts.getView(), this.opts.count);
  }

  update(nowMs: number): void {
    this.now = nowMs;
    if (this.last === null) {
      this.last = this.current();
      this.lastWriteAt = nowMs;
      return;
    }
    if (this.paused || nowMs - this.lastWriteAt < this.intervalMs) return;
    this.write();
  }

  /** Write the current link now if it differs from the last one written. */
  flush(): void {
    if (this.last === null) this.last = "";
    this.write();
  }

  /** Hold the timed writes, for the intro tour, whose mid-flight camera is not a link worth keeping. */
  pause(paused: boolean): void {
    this.paused = paused;
  }

  private write(): void {
    const next = this.current();
    if (next === this.last) return;
    this.last = next;
    this.lastWriteAt = this.now;
    this.opts.write(next);
  }
}
