import { describe, expect, it } from "vitest";
import { shotFilename, shotPixelRatio } from "./screenshot";

describe("shotPixelRatio", () => {
  it("renders at twice the CSS size", () => {
    expect(shotPixelRatio(1280)).toBe(2);
    expect(shotPixelRatio(1600)).toBe(2);
  });

  it("caps the image at 3200 px wide", () => {
    expect(shotPixelRatio(2560)).toBeCloseTo(1.25, 9);
    expect(shotPixelRatio(2560) * 2560).toBeCloseTo(3200, 6);
  });

  it("falls back to 2 for a zero-width canvas instead of dividing by zero", () => {
    expect(shotPixelRatio(0)).toBe(2);
  });
});

describe("shotFilename", () => {
  it("names the selected cell by its root id, else the volume", () => {
    expect(shotFilename("864691135615535520")).toBe("one-cubic-millimetre-864691135615535520.png");
    expect(shotFilename(null)).toBe("one-cubic-millimetre-volume.png");
  });
});
