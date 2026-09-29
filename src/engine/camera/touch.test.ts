import { describe, expect, it } from "vitest";
import { stickVector } from "./touch";

describe("stickVector", () => {
  it("is zero inside the dead zone and for a zero radius", () => {
    expect(stickVector(3, 2, 60)).toEqual({ x: 0, y: 0 });
    expect(stickVector(50, 50, 0)).toEqual({ x: 0, y: 0 });
  });

  it("reaches exactly 1 at the radius and never beyond it", () => {
    const edge = stickVector(60, 0, 60);
    expect(edge.x).toBeCloseTo(1, 9);
    const past = stickVector(600, 800, 60);
    expect(Math.hypot(past.x, past.y)).toBeCloseTo(1, 9);
  });

  it("points up on screen as positive y and right as positive x", () => {
    const up = stickVector(0, -60, 60);
    expect(up.y).toBeCloseTo(1, 9);
    expect(up.x).toBeCloseTo(0, 9);
    const right = stickVector(30, 0, 60);
    expect(right.x).toBeGreaterThan(0);
    expect(right.y).toBeCloseTo(0, 9);
  });

  it("rises monotonically with deflection", () => {
    let prev = 0;
    for (let d = 10; d <= 60; d += 5) {
      const m = stickVector(d, 0, 60).x;
      expect(m).toBeGreaterThanOrEqual(prev);
      prev = m;
    }
  });
});
