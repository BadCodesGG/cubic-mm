/**
 * High-resolution screenshots of the volume. The canvas holds only the scene, so the HUD (DOM)
 * is never in the picture.
 *
 * A WebGPU canvas's contents are only guaranteed until the frame is presented, at the end of the
 * task that drew it. So the capture renders its own frame at the higher pixel ratio and calls
 * `toBlob` in the same task, before anything can present or resize the canvas, then restores the
 * ratio and draws a normal frame so the visitor never sees an empty canvas.
 */

/** Longest edge of a screenshot, px. */
export const SHOT_MAX_WIDTH = 3200;

/** Pixel ratio for a screenshot of a canvas `cssWidth` CSS px wide: 2, capped at 3200 px wide. */
export function shotPixelRatio(cssWidth: number): number {
  return cssWidth > 0 ? Math.min(2, SHOT_MAX_WIDTH / cssWidth) : 2;
}

export function shotFilename(rootId: string | null): string {
  return `one-cubic-millimetre-${rootId ?? "volume"}.png`;
}

export interface CaptureTarget {
  canvas: HTMLCanvasElement;
  getPixelRatio(): number;
  setPixelRatio(ratio: number): void;
  /** Re-applies the drawing-buffer size and the pixel-size uniforms (the app's `resize`). */
  resize(): void;
  /** Draws one frame (the post pipeline). */
  render(): void;
}

/** Renders one high-resolution frame and encodes it as PNG. Resolves null if the browser could not encode it. */
export function captureFrame(target: CaptureTarget): Promise<Blob | null> {
  const { canvas } = target;
  const previous = target.getPixelRatio();
  let png: Promise<Blob | null> = Promise.resolve(null);
  try {
    target.setPixelRatio(shotPixelRatio(canvas.clientWidth));
    target.resize();
    target.render();
    // toBlob snapshots the bitmap synchronously; only the encoding is asynchronous.
    png = new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  } finally {
    target.setPixelRatio(previous);
    target.resize();
    target.render();
  }
  return png;
}

/** Hands `blob` to the browser as a file download. */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Revoking at once can cancel the download in some browsers.
  window.setTimeout(() => URL.revokeObjectURL(url), 10_000);
}
