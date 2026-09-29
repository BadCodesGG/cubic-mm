import { describe, expect, it } from "vitest";
import { POSITION_SCALE, tanhCurve } from "./voices";

describe("tanhCurve", () => {
  const curve = tanhCurve();

  it("is bounded by full scale and hits it at the ends", () => {
    expect(curve[0]).toBeCloseTo(-1, 6);
    expect(curve[curve.length - 1]).toBeCloseTo(1, 6);
    for (const v of curve) expect(Math.abs(v)).toBeLessThanOrEqual(1 + 1e-6);
  });

  it("is monotonic, odd-symmetric and passes silence through", () => {
    for (let i = 1; i < curve.length; i++) expect(curve[i]).toBeGreaterThan(curve[i - 1]);
    for (let i = 0; i < curve.length; i++) expect(curve[i]).toBeCloseTo(-curve[curve.length - 1 - i], 6);
  });
});

describe("POSITION_SCALE", () => {
  it("makes 100 µm one audio unit", () => {
    expect(100 * POSITION_SCALE).toBe(1);
  });
});
