/**
 * Dual-stick touch input for phones: the left stick moves (up is forward), the right stick looks.
 * The stick maths lives here so it is tested without a DOM; `components/hud/touch-pad.tsx` owns
 * the elements and feeds it pointer offsets.
 */

export interface Stick {
  /** Right is positive. */
  x: number;
  /** Up (away from the visitor, on screen) is positive. */
  y: number;
}

export const NO_STICK: Readonly<Stick> = { x: 0, y: 0 };

/**
 * A finger offset from the stick's centre (CSS px, y down as on screen) as a unit-disc vector: the
 * dead zone reads as zero, the rest is rescaled so the response starts at zero at the dead-zone edge
 * and reaches 1 at `radius`, and a light curve gives fine control near the centre.
 */
export function stickVector(dx: number, dy: number, radius: number, deadZone = 0.14): Stick {
  const len = Math.hypot(dx, dy);
  if (radius <= 0 || len <= deadZone * radius) return { x: 0, y: 0 };
  const t = Math.min(1, (len - deadZone * radius) / (radius * (1 - deadZone)));
  const shaped = t * t * (3 - 2 * t) * 0.35 + t * 0.65;
  return { x: (dx / len) * shaped, y: (-dy / len) * shaped };
}

/** Radians per second of yaw or pitch at full right-stick deflection. */
export const TOUCH_LOOK_RATE = 1.9;
