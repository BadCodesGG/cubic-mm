import { Color, NoToneMapping, WebGPURenderer } from "three/webgpu";

/** Near-black with a blue cast: the colour of the space between the cells. */
export const BACKGROUND = new Color(0x04060b);

export interface RendererHandle {
  renderer: WebGPURenderer;
  /** False when the browser had no WebGPU and three fell back to its WebGL2 backend. */
  isWebGPU: boolean;
}

export async function createRenderer(
  canvas: HTMLCanvasElement,
  { forceWebGL = false, trackTimestamp = false }: { forceWebGL?: boolean; trackTimestamp?: boolean } = {},
): Promise<RendererHandle> {
  // No MSAA: every ribbon and soma already has a soft analytic edge, and 4x MSAA on the
  // half-float scene target was a quarter to two thirds of the GPU time (blend bandwidth).
  const renderer = new WebGPURenderer({ canvas, antialias: false, forceWebGL, powerPreference: "high-performance", trackTimestamp });
  // The app lowers this once it knows the quality level (see `maxPixelRatio` in quality.ts).
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(BACKGROUND, 1);
  renderer.toneMapping = NoToneMapping;
  await renderer.init();
  const backend = renderer.backend as { isWebGPUBackend?: boolean };
  return { renderer, isWebGPU: backend.isWebGPUBackend === true };
}
