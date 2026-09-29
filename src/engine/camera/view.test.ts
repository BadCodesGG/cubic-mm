import { describe, expect, it } from "vitest";
import { PerspectiveCamera, Vector3 } from "three/webgpu";
import { WORLD_UP } from "./fly";
import { createHudFrame, niceScale, pxPerUmAt, updateHudFrame } from "./view";

function cameraAt(x: number, y: number, z: number, lookAt: [number, number, number]) {
  const c = new PerspectiveCamera(55, 16 / 9, 0.5, 4000);
  c.up.copy(WORLD_UP);
  c.position.set(x, y, z);
  c.lookAt(...lookAt);
  c.updateMatrixWorld();
  return c;
}

describe("niceScale", () => {
  it("keeps 100 µm while it fits between 40 and 240 px", () => {
    expect(niceScale(1)).toEqual({ um: 100, px: 100 });
    expect(niceScale(0.4).um).toBe(100);
    expect(niceScale(2.4).um).toBe(100);
  });

  it("switches to a round length whose bar still fits", () => {
    const close = niceScale(6); // 100 µm would be 600 px
    expect(close.um).toBe(20);
    expect(close.px).toBeCloseTo(120, 9);
    const far = niceScale(0.05); // 100 µm would be 5 px
    expect(far.um).toBe(1000);
    expect(far.px).toBeCloseTo(50, 9);
  });

  it("has no bar for a non-positive magnification", () => {
    expect(niceScale(0).px).toBe(0);
  });
});

describe("pxPerUmAt", () => {
  it("is projection scale * half the height / depth", () => {
    const c = cameraAt(0, 0, 0, [0, 0, 1]);
    const f = 1 / Math.tan((55 * Math.PI) / 360);
    expect(pxPerUmAt(200, c, 900)).toBeCloseTo((f * 450) / 200, 6);
  });
});

describe("updateHudFrame", () => {
  it("measures the scale at the focus depth and points the compass at the pia", () => {
    // Looking along +z with y down on the world: the pia (low y) is toward the top of the screen.
    const c = cameraAt(0, 0, 0, [0, 0, 1]);
    const frame = createHudFrame();
    updateHudFrame(frame, c, 1600, 900, new Vector3(0, 0, 400), null);
    const f = 1 / Math.tan((55 * Math.PI) / 360);
    expect(frame.scaleUm * (frame.scalePx / frame.scaleUm)).toBeCloseTo(frame.scalePx, 9);
    expect(frame.scalePx / frame.scaleUm).toBeCloseTo((f * 450) / 400, 6);
    expect(frame.piaAngle).toBeCloseTo(0, 6);
    expect(frame.piaLength).toBeCloseTo(1, 6);
  });

  it("rolls the compass with the camera and shortens it looking straight down the depth axis", () => {
    const side = cameraAt(0, 0, 0, [1, 0, 0]);
    const frame = createHudFrame();
    updateHudFrame(frame, side, 1600, 900, null, null);
    expect(Math.abs(frame.piaAngle)).toBeCloseTo(0, 6);
    const down = cameraAt(0, 0, 0, [0, 1, 0]);
    down.up.set(0, 0, 1);
    down.lookAt(0, 1, 0);
    down.updateMatrixWorld();
    updateHudFrame(frame, down, 1600, 900, null, null);
    expect(frame.piaLength).toBeLessThan(0.01);
  });

  it("projects the selected soma to canvas pixels, centre for a soma dead ahead", () => {
    const c = cameraAt(0, 0, 0, [0, 0, 1]);
    const frame = createHudFrame();
    updateHudFrame(frame, c, 1600, 900, null, new Vector3(0, 0, 300));
    expect(frame.selectedScreen![0]).toBeCloseTo(800, 3);
    expect(frame.selectedScreen![1]).toBeCloseTo(450, 3);
    expect(frame.selectedDistanceUm).toBeCloseTo(300, 6);
    // A soma above the view axis (lower y is up on screen) lands in the top half.
    updateHudFrame(frame, c, 1600, 900, null, new Vector3(0, -50, 300));
    expect(frame.selectedScreen![1]).toBeLessThan(450);
  });

  it("hides the ring for a soma behind the camera or no selection", () => {
    const c = cameraAt(0, 0, 0, [0, 0, 1]);
    const frame = createHudFrame();
    updateHudFrame(frame, c, 1600, 900, null, new Vector3(0, 0, -300));
    expect(frame.selectedScreen).toBeNull();
    updateHudFrame(frame, c, 1600, 900, null, null);
    expect(frame.selectedScreen).toBeNull();
    expect(frame.selectedDistanceUm).toBe(0);
  });
});
