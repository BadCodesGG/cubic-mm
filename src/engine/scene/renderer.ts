import { Color, NoToneMapping, WebGPURenderer } from "three/webgpu";

/** Near-black with a blue cast: the colour of the space between the cells. */
export const BACKGROUND = new Color(0x04060b);

export interface RendererHandle {
  renderer: WebGPURenderer;
  /** False when the browser had no WebGPU and three fell back to its WebGL2 backend. */
  isWebGPU: boolean;
}

function isMobile(): boolean {
  return /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent);
}

export async function createRenderer(
  canvas: HTMLCanvasElement,
  { forceWebGL = false }: { forceWebGL?: boolean } = {},
): Promise<RendererHandle> {
  const renderer = new WebGPURenderer({ canvas, antialias: true, forceWebGL, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, isMobile() ? 1.5 : 2));
  renderer.setClearColor(BACKGROUND, 1);
  renderer.toneMapping = NoToneMapping;
  await renderer.init();
  const backend = renderer.backend as { isWebGPUBackend?: boolean };
  return { renderer, isWebGPU: backend.isWebGPUBackend === true };
}
