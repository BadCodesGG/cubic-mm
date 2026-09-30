/**
 * A cell's wiring in the scene: one straight line from its soma to each partner soma, plus a faint
 * glow on the partner somas. Outputs are warm amber, inputs cool cyan, inputs from inhibitory cells
 * violet; a line is brighter the more synapses the pair share (log scale) and at most 400 are drawn,
 * largest first.
 *
 * Built like the cascade lines (`scene/cascade.ts`): geometry rewritten from the CPU, colours baked
 * into the vertices, additive and depth-write-free so a hub's lines add up like light. A graph
 * fades in over 150 ms and out over 250 ms (`approach`), and a graph that is replaced keeps
 * fading out while the new one comes in, so crossing the cluster with the mouse does not strobe.
 * The hovered graph and the pinned graph are separate entries; the pinned one dims while another
 * cell is hovered.
 */

import {
  AdditiveBlending,
  BufferAttribute,
  BufferGeometry,
  DynamicDrawUsage,
  Group,
  InstancedBufferAttribute,
  InstancedBufferGeometry,
  Float32BufferAttribute,
  LineBasicNodeMaterial,
  LineSegments,
  Mesh,
  MeshBasicNodeMaterial,
} from "three/webgpu";
import { attribute, cameraProjectionMatrix, exp, float, max, min, modelViewMatrix, positionLocal, positionView, smoothstep, varying, vec3, vec4, vertexColor } from "three/tsl";
import type { Dataset } from "../data";
import { LineKind, MAX_LINES, PIN_DIM, WiringState, approach, lineWeight, wiringLines, type PartnerIndex, type WiringLine } from "../partners";
import type { SceneUniforms } from "./uniforms";

/** Graphs alive at once: the pinned one, the hovered one, and a few still fading out. */
const MAX_GRAPHS = 5;
/** rgb of each line kind. */
const COLOR: Record<LineKind, readonly [number, number, number]> = {
  [LineKind.Output]: [1.0, 0.5, 0.1],
  [LineKind.Input]: [0.1, 0.72, 1.0],
  [LineKind.InhibitoryInput]: [0.68, 0.38, 1.0],
};
/** Brightness of a line's peak, before its own weight and the graph's level. */
const LINE_GAIN = 2.4;
/** The far end of a line is this much of the near end, so a line reads as leaving one cell. */
const FAR_END = 0.6;
/** Lines that meet at one soma add up like light, so a busy graph draws each line dimmer: full up to this many lines. */
const CROWD = 50;
const SPRITE_GAIN = 1.3;
const SPRITE_UM = 3.2;
const ORIGIN_SPRITE_UM = 4.5;

interface Graph {
  neuron: number;
  /** "hover" or "pin" while it is the wanted one; null once it is only fading out. */
  slot: "hover" | "pin" | null;
  level: number;
  lines: WiringLine[];
}

export interface PartnerGraph {
  group: Group;
  /** Advances the fades by `dtS` (real seconds) and rewrites the geometry when something changed. */
  update(dtS: number): void;
  /** Lines being drawn right now. */
  readonly lineCount: number;
  dispose(): void;
}

export function createPartnerGraph(data: Dataset, index: PartnerIndex, state: WiringState, u: SceneUniforms): PartnerGraph {
  const { somaUm, inhibitory } = data.neurons;
  const maxLines = MAX_GRAPHS * MAX_LINES;
  const maxSprites = MAX_GRAPHS * (MAX_LINES + 1);

  // Lines.
  const position = new BufferAttribute(new Float32Array(maxLines * 6), 3);
  const color = new BufferAttribute(new Float32Array(maxLines * 6), 3);
  position.setUsage(DynamicDrawUsage);
  color.setUsage(DynamicDrawUsage);
  const lineGeometry = new BufferGeometry();
  lineGeometry.setAttribute("position", position);
  lineGeometry.setAttribute("color", color);
  lineGeometry.setDrawRange(0, 0);
  const lineMaterial = new LineBasicNodeMaterial();
  // Lines fade with view depth a third as fast as the ribbons, as the cascade's do.
  const viewDepth = positionView.z.negate();
  lineMaterial.colorNode = vec4(vertexColor().rgb.mul(exp(viewDepth.div(u.hazeDistance.mul(3)).negate())), 1);
  lineMaterial.transparent = true;
  lineMaterial.depthWrite = false;
  lineMaterial.blending = AdditiveBlending;
  const lines = new LineSegments(lineGeometry, lineMaterial);
  lines.frustumCulled = false;
  lines.renderOrder = 5;
  lines.visible = false;

  // Partner glow: one camera-facing sprite per partner soma, and a larger one on the cell itself.
  const centre = new InstancedBufferAttribute(new Float32Array(maxSprites * 4), 4);
  const glow = new InstancedBufferAttribute(new Float32Array(maxSprites * 3), 3);
  centre.setUsage(DynamicDrawUsage);
  glow.setUsage(DynamicDrawUsage);
  const spriteGeometry = new InstancedBufferGeometry();
  spriteGeometry.setAttribute("position", new Float32BufferAttribute([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0], 3));
  spriteGeometry.setIndex([0, 1, 2, 0, 2, 3]);
  spriteGeometry.setAttribute("aCentre", centre);
  spriteGeometry.setAttribute("aGlow", glow);
  spriteGeometry.instanceCount = 0;
  const sprites = new Mesh(spriteGeometry, spriteMaterial(u));
  sprites.frustumCulled = false;
  sprites.renderOrder = 5;
  sprites.visible = false;

  const group = new Group();
  group.add(lines, sprites);

  const pos = position.array as Float32Array;
  const col = color.array as Float32Array;
  const cen = centre.array as Float32Array;
  const glw = glow.array as Float32Array;
  let graphs: Graph[] = [];
  let drawn = 0;
  let lastKey = "";

  const open = (neuron: number, slot: "hover" | "pin"): Graph => ({ neuron, slot, level: 0, lines: wiringLines(index, inhibitory, neuron) });

  /** Make the entries match what is wanted: a wanted cell with no live entry gets one, an unwanted live entry starts fading out. */
  const reconcile = () => {
    const want = state.shown();
    for (const slot of ["hover", "pin"] as const) {
      const neuron = want[slot];
      const live = graphs.find((g) => g.slot === slot);
      if (live && live.neuron === neuron) continue;
      if (live) live.slot = null;
      if (neuron >= 0) graphs.push(open(neuron, slot));
    }
    // Too many still fading: drop the faintest.
    while (graphs.length > MAX_GRAPHS) {
      let faint = -1;
      for (let i = 0; i < graphs.length; i++) if (graphs[i].slot === null && (faint < 0 || graphs[i].level < graphs[faint].level)) faint = i;
      if (faint < 0) break;
      graphs.splice(faint, 1);
    }
    return want;
  };

  const write = () => {
    let n = 0;
    let s = 0;
    for (const g of graphs) {
      if (g.level <= 0) continue;
      const crowd = Math.min(1, Math.pow(CROWD / Math.max(1, g.lines.length), 0.4));
      const ox = somaUm[g.neuron * 3];
      const oy = somaUm[g.neuron * 3 + 1];
      const oz = somaUm[g.neuron * 3 + 2];
      // The cell itself: a soft warm-white glow.
      const o = s * 4;
      cen[o] = ox;
      cen[o + 1] = oy;
      cen[o + 2] = oz;
      cen[o + 3] = ORIGIN_SPRITE_UM;
      glw[s * 3] = glw[s * 3 + 1] = glw[s * 3 + 2] = SPRITE_GAIN * 0.8 * g.level;
      s++;
      for (const line of g.lines) {
        const [r, gr, b] = COLOR[line.kind];
        const k = LINE_GAIN * crowd * g.level * lineWeight(line.synapses);
        const i = n * 6;
        const px = somaUm[line.neuron * 3];
        const py = somaUm[line.neuron * 3 + 1];
        const pz = somaUm[line.neuron * 3 + 2];
        pos[i] = ox;
        pos[i + 1] = oy;
        pos[i + 2] = oz;
        pos[i + 3] = px;
        pos[i + 4] = py;
        pos[i + 5] = pz;
        col[i] = r * k;
        col[i + 1] = gr * k;
        col[i + 2] = b * k;
        col[i + 3] = r * k * FAR_END;
        col[i + 4] = gr * k * FAR_END;
        col[i + 5] = b * k * FAR_END;
        n++;
        const j = s * 4;
        cen[j] = px;
        cen[j + 1] = py;
        cen[j + 2] = pz;
        cen[j + 3] = SPRITE_UM;
        const sk = SPRITE_GAIN * g.level * lineWeight(line.synapses);
        glw[s * 3] = r * sk;
        glw[s * 3 + 1] = gr * sk;
        glw[s * 3 + 2] = b * sk;
        s++;
      }
    }
    drawn = n;
    lines.visible = n > 0;
    sprites.visible = s > 0;
    lineGeometry.setDrawRange(0, n * 2);
    spriteGeometry.instanceCount = s;
    position.needsUpdate = color.needsUpdate = centre.needsUpdate = glow.needsUpdate = true;
  };

  return {
    group,
    update(dtS) {
      const want = reconcile();
      const hovering = want.hover >= 0;
      for (const g of graphs) {
        // The pinned graph steps back while another cell is hovered.
        const target = g.slot === null ? 0 : g.slot === "pin" && hovering ? PIN_DIM : 1;
        g.level = approach(g.level, target, dtS);
      }
      graphs = graphs.filter((g) => g.slot !== null || g.level > 0);
      // Nothing showing and nothing drawn last frame: leave the buffers alone.
      if (graphs.length === 0 && !lines.visible) return;
      const key = graphs.map((g) => `${g.neuron}:${g.level.toFixed(3)}`).join(",");
      if (key === lastKey && graphs.length > 0) return;
      lastKey = key;
      write();
    },
    get lineCount() {
      return drawn;
    },
    dispose() {
      lineGeometry.dispose();
      lineMaterial.dispose();
      spriteGeometry.dispose();
      (sprites.material as MeshBasicNodeMaterial).dispose();
    },
  };
}

function spriteMaterial(u: SceneUniforms): MeshBasicNodeMaterial {
  const aCentre = attribute("aCentre", "vec4");
  const aGlow = attribute("aGlow", "vec3");
  const centreView = modelViewMatrix.mul(vec4(aCentre.xyz, 1)).xyz;
  const depth = max(centreView.z.negate(), 0.5);
  const pxPerUm = u.pixelScale.div(depth);
  // At least 3 px so a far partner still reads, at most glowCapPx so a near one is not a blob.
  const radius = min(max(aCentre.w, float(3).div(pxPerUm)), u.glowCapPx.div(pxPerUm));
  const corner = positionLocal.xy;
  const material = new MeshBasicNodeMaterial();
  material.vertexNode = cameraProjectionMatrix.mul(vec4(centreView.add(vec3(corner.mul(radius), 0)), 1));
  const light = exp(depth.div(u.hazeDistance.mul(3)).negate()).mul(smoothstep(2, 20, depth));
  const vCorner = varying(corner);
  const vLight = varying(aGlow.mul(light));
  material.colorNode = vec4(vLight.mul(exp(vCorner.dot(vCorner).mul(-4))), 1);
  material.transparent = true;
  material.depthWrite = false;
  material.blending = AdditiveBlending;
  return material;
}
