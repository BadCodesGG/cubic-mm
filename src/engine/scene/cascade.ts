/**
 * The cascade in the scene: one straight line from the soma that drove a cell to the soma that
 * fired, drawn the moment it fires and gone 2.5 s later. Warm amber for hop 1, dimmer for hop 2.
 *
 * The 400-segment geometry is rewritten from the tracker every frame (a few thousand floats), with
 * the fade baked into the vertex colours, so the material is a plain `LineBasicNodeMaterial` that
 * behaves the same on both backends. Additive and depth-write-free, so overlapping lines add up
 * like light and bloom picks up the fresh ones. Lines are 1 px on both backends, as WebGL2 has no
 * wide lines and WebGPU has no line width at all.
 */

import { AdditiveBlending, BufferAttribute, BufferGeometry, DynamicDrawUsage, LineBasicNodeMaterial, LineSegments } from "three/webgpu";
import { exp, positionView, vec4, vertexColor } from "three/tsl";
import type { CascadeTracker } from "../cascade";
import { MAX_EDGES } from "../cascade";
import type { Dataset } from "../data";
import type { SceneUniforms } from "./uniforms";

/** Seconds a line takes to fade out after its target fires. */
export const FADE_S = 2.5;
const HOP1 = [1.0, 0.58, 0.2] as const;
const HOP2 = [0.62, 0.36, 0.16] as const;
const HOP1_GAIN = 3.5;
const HOP2_GAIN = 1.8;

export interface CascadeLines {
  object: LineSegments;
  /** Rewrites the lines for simulation time `time`. Cheap when nothing is showing. */
  update(time: number): void;
  dispose(): void;
}

export function createCascadeLines(data: Dataset, tracker: CascadeTracker, u: SceneUniforms): CascadeLines {
  const { somaUm } = data.neurons;
  const position = new BufferAttribute(new Float32Array(MAX_EDGES * 6), 3);
  const color = new BufferAttribute(new Float32Array(MAX_EDGES * 6), 3);
  position.setUsage(DynamicDrawUsage);
  color.setUsage(DynamicDrawUsage);
  const geometry = new BufferGeometry();
  geometry.setAttribute("position", position);
  geometry.setAttribute("color", color);
  geometry.setDrawRange(0, 0);

  const material = new LineBasicNodeMaterial();
  const depth = positionView.z.negate();
  material.colorNode = vec4(vertexColor().rgb.mul(exp(depth.div(u.hazeDistance.mul(3)).negate())), 1);
  material.transparent = true;
  material.depthWrite = false;
  material.blending = AdditiveBlending;

  const object = new LineSegments(geometry, material);
  object.frustumCulled = false;
  object.renderOrder = 5;
  object.visible = false;

  const pos = position.array as Float32Array;
  const col = color.array as Float32Array;
  return {
    object,
    update(time) {
      let n = 0;
      for (const [from, to, born] of tracker.edges()) {
        const age = time - born;
        if (age < 0 || age >= FADE_S) continue;
        const k = 1 - age / FADE_S;
        const [r, g, b] = from === tracker.rootNeuron ? HOP1 : HOP2;
        const s = k * k * (from === tracker.rootNeuron ? HOP1_GAIN : HOP2_GAIN);
        const o = n * 6;
        pos[o] = somaUm[from * 3];
        pos[o + 1] = somaUm[from * 3 + 1];
        pos[o + 2] = somaUm[from * 3 + 2];
        pos[o + 3] = somaUm[to * 3];
        pos[o + 4] = somaUm[to * 3 + 1];
        pos[o + 5] = somaUm[to * 3 + 2];
        col[o] = col[o + 3] = r * s;
        col[o + 1] = col[o + 4] = g * s;
        col[o + 2] = col[o + 5] = b * s;
        n++;
      }
      // Nothing to draw and nothing drawn last frame: leave the buffers alone.
      if (n === 0 && !object.visible) return;
      object.visible = n > 0;
      geometry.setDrawRange(0, n * 2);
      position.needsUpdate = true;
      color.needsUpdate = true;
    },
    dispose() {
      geometry.dispose();
      material.dispose();
    },
  };
}
