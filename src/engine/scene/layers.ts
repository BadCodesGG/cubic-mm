/**
 * Six barely-visible horizontal sheets at the cortical layer boundaries: the pia and the five
 * internal boundaries. y is depth (pia at low y), so each sheet is an xz plane.
 *
 * The dataset rarely has somas in every layer, so the boundaries are not read off directly.
 * Each layer's median soma depth is matched to that layer's centre in a canonical mouse visual
 * cortex profile, and a least-squares fit (weighted by cell count) gives the pia depth and the
 * cortical thickness, which place all six sheets.
 */

import { AdditiveBlending, DoubleSide, Mesh, MeshBasicNodeMaterial, PlaneGeometry, Group } from "three/webgpu";
import { exp, float, min, positionView, smoothstep, uv, vec3, vec4 } from "three/tsl";
import type { Dataset } from "../data";
import type { SceneUniforms } from "./uniforms";

/** Relative depth of each layer edge, pia = 0 to white matter = 1 (L1, L2, L3, L4, L5, L6). */
const EDGES = [0, 0.1, 0.2, 0.36, 0.5, 0.72, 1];
const FALLBACK_THICKNESS_UM = 900;

function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  return s[s.length >> 1];
}

/** y of the pia and the five internal layer boundaries, pia first. */
export function layerBoundaries(data: Dataset): number[] {
  const byLayer = new Map<number, number[]>();
  const { somaUm, layer, count } = data.neurons;
  for (let i = 0; i < count; i++) {
    const l = layer[i];
    if (l < 1 || l > 6) continue;
    if (!byLayer.has(l)) byLayer.set(l, []);
    byLayer.get(l)!.push(somaUm[i * 3 + 1]);
  }
  if (byLayer.size === 0) return [];
  // Weighted least squares of median depth against canonical layer centre: y = pia + thickness * c.
  let sw = 0;
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  for (const [l, ys] of byLayer) {
    const w = ys.length;
    const x = (EDGES[l - 1] + EDGES[l]) / 2;
    const y = median(ys);
    sw += w;
    sx += w * x;
    sy += w * y;
    sxx += w * x * x;
    sxy += w * x * y;
  }
  const denom = sw * sxx - sx * sx;
  let thickness = denom > 1e-9 ? (sw * sxy - sx * sy) / denom : 0;
  if (!(thickness > 300 && thickness < 3000)) thickness = FALLBACK_THICKNESS_UM;
  const pia = (sy - thickness * sx) / sw;
  return EDGES.slice(0, 6).map((e) => pia + thickness * e);
}

export function createLayers(data: Dataset, u: SceneUniforms): { group: Group; dispose(): void } {
  const { min: bmin, max: bmax } = data.manifest.boundsUm;
  const sizeX = bmax[0] - bmin[0];
  const sizeZ = bmax[2] - bmin[2];
  const geometry = new PlaneGeometry(sizeX, sizeZ);
  geometry.rotateX(-Math.PI / 2);

  const edge = min(
    min(smoothstep(0, 0.2, uv().x), smoothstep(1, 0.8, uv().x)),
    min(smoothstep(0, 0.2, uv().y), smoothstep(1, 0.8, uv().y)),
  );
  const depth = positionView.z.negate();
  const material = new MeshBasicNodeMaterial();
  material.colorNode = vec4(
    vec3(0.2, 0.3, 0.5)
      .mul(0.012)
      .mul(edge)
      .mul(exp(depth.div(u.hazeDistance.mul(1.4)).negate()))
      .mul(smoothstep(float(10), float(60), depth)),
    1,
  );
  material.transparent = true;
  material.depthWrite = false;
  material.blending = AdditiveBlending;
  material.side = DoubleSide;

  const group = new Group();
  for (const y of layerBoundaries(data)) {
    const plane = new Mesh(geometry, material);
    plane.position.set((bmin[0] + bmax[0]) / 2, y, (bmin[2] + bmax[2]) / 2);
    plane.frustumCulled = false;
    group.add(plane);
  }
  return {
    group,
    dispose() {
      geometry.dispose();
      material.dispose();
    },
  };
}
