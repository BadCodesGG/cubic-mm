import { describe, expect, it } from "vitest";
import { mapHeading, mapTransform, nearestSoma } from "./minimap";

// A 1000 x 500 µm footprint (x by z) on a 140 px map with 10 px padding: 120 px across, 0.12 px/µm.
const bounds = { min: [0, 0, 0] as [number, number, number], max: [1000, 800, 500] as [number, number, number] };

describe("mapTransform", () => {
  const map = mapTransform(bounds, 140, 10);

  it("fits the x-z footprint inside the padding, keeps its aspect, and centres it", () => {
    expect(map.scale).toBeCloseTo(0.12, 9);
    expect(map.rect).toEqual({ x: 10, y: 40, w: 120, h: 60 });
  });

  it("puts +x to the right and +z at the top, as seen looking down from the pia", () => {
    expect(map.toMap(0, 0)).toEqual([10, 100]);
    expect(map.toMap(1000, 500)).toEqual([130, 40]);
    expect(map.toMap(500, 250)).toEqual([70, 70]);
  });
});

describe("mapHeading", () => {
  it("points up for a camera facing +z and right for one facing +x", () => {
    const [ux, uy] = mapHeading(0);
    expect(ux).toBeCloseTo(0, 9);
    expect(uy).toBeCloseTo(-1, 9);
    const [rx, ry] = mapHeading(Math.PI / 2);
    expect(rx).toBeCloseTo(1, 9);
    expect(ry).toBeCloseTo(0, 9);
  });
});

describe("nearestSoma", () => {
  const map = mapTransform(bounds, 140, 10);
  // Somas at map (70,70), (82,70) and (10,100); y (depth) never matters on the map.
  const somas = new Float32Array([500, 10, 250, 600, 700, 250, 0, 300, 0]);

  it("returns the soma nearest the click", () => {
    expect(nearestSoma(somas, map, 71, 70)).toBe(0);
    expect(nearestSoma(somas, map, 80, 71)).toBe(1);
    expect(nearestSoma(somas, map, 12, 99)).toBe(2);
  });

  it("returns -1 when nothing is within 12 px", () => {
    expect(nearestSoma(somas, map, 40, 20)).toBe(-1);
    expect(nearestSoma(somas, map, 70, 82.5)).toBe(-1);
    expect(nearestSoma(somas, map, 70, 81.5)).toBe(0);
  });
});
