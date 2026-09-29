/**
 * Geometry for the HUD minimap: the volume's x-z footprint seen from the pia looking down, so +x
 * is to the right and +z to the top (that makes a turn to the right a clockwise turn on the map).
 * Pure, so the HUD component only draws.
 */

type Vec3 = readonly [number, number, number];

export interface MapTransform {
  /** Pixels per µm. */
  scale: number;
  /** The footprint's rectangle on the map, px. */
  rect: { x: number; y: number; w: number; h: number };
  /** Map pixel of world (x, z). */
  toMap(x: number, z: number): [number, number];
}

export function mapTransform(bounds: { min: Vec3; max: Vec3 }, sizePx: number, padPx: number): MapTransform {
  const [x0, , z0] = bounds.min;
  const [x1, , z1] = bounds.max;
  const inner = sizePx - 2 * padPx;
  const scale = inner / Math.max(x1 - x0, z1 - z0, 1e-6);
  const w = (x1 - x0) * scale;
  const h = (z1 - z0) * scale;
  const left = (sizePx - w) / 2;
  const top = (sizePx - h) / 2;
  return {
    scale,
    rect: { x: left, y: top, w, h },
    toMap: (x, z) => [left + (x - x0) * scale, top + (z1 - z) * scale],
  };
}

/** Unit vector on the map (canvas y down) for a camera yaw of `atan2(dir.x, dir.z)`. */
export function mapHeading(yaw: number): [number, number] {
  return [Math.sin(yaw), -Math.cos(yaw)];
}

/** Index of the soma drawn nearest map pixel (px, py), or -1 if none is within `maxPx`. */
export function nearestSoma(somaUm: Float32Array, map: MapTransform, px: number, py: number, maxPx = 12): number {
  let best = -1;
  let bestD = maxPx * maxPx;
  for (let i = 0; i * 3 < somaUm.length; i++) {
    const [mx, my] = map.toMap(somaUm[i * 3], somaUm[i * 3 + 2]);
    const d = (mx - px) ** 2 + (my - py) ** 2;
    if (d <= bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}
