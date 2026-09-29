import { describe, expect, it } from "vitest";
import { FrameBudget, maxPixelRatio, resolveQuality, stepDown, type QualityInputs } from "./quality";

const desktop: QualityInputs = { pref: "auto", isWebGPU: true, coarsePointer: false, viewportWidth: 1600 };

describe("resolveQuality", () => {
  it("starts a desktop WebGPU visitor on auto at hi with 2 pieces, and measures", () => {
    expect(resolveQuality(desktop)).toEqual({ level: { tier: "hi", pieces: 2 }, adaptive: true });
  });

  it("puts WebGL2 on lite with 2 pieces", () => {
    expect(resolveQuality({ ...desktop, isWebGPU: false }).level).toEqual({ tier: "lite", pieces: 2 });
  });

  it("puts a coarse pointer or a viewport narrower than 900 px on lite", () => {
    expect(resolveQuality({ ...desktop, coarsePointer: true }).level.tier).toBe("lite");
    expect(resolveQuality({ ...desktop, viewportWidth: 899 }).level.tier).toBe("lite");
    expect(resolveQuality({ ...desktop, viewportWidth: 900 }).level.tier).toBe("hi");
  });

  it("honours an explicit preference and never adapts it", () => {
    expect(resolveQuality({ ...desktop, pref: "hi", coarsePointer: true })).toEqual({
      level: { tier: "hi", pieces: 3 },
      adaptive: false,
    });
    expect(resolveQuality({ ...desktop, pref: "lite" })).toEqual({ level: { tier: "lite", pieces: 2 }, adaptive: false });
  });

  it("lets a URL override pin the tier and the piece count, without adapting", () => {
    expect(resolveQuality({ ...desktop, override: { tier: "hi", pieces: 3 } })).toEqual({
      level: { tier: "hi", pieces: 3 },
      adaptive: false,
    });
    expect(resolveQuality({ ...desktop, override: { pieces: 1 } })).toEqual({ level: { tier: "hi", pieces: 1 }, adaptive: false });
  });

  it("never adapts a scripted run", () => {
    expect(resolveQuality({ ...desktop, scripted: true }).adaptive).toBe(false);
  });
});

describe("stepDown", () => {
  it("walks hi/3 and hi/2 to lite/2, then lite/1, then stops", () => {
    expect(stepDown({ tier: "hi", pieces: 3 })).toEqual({ tier: "hi", pieces: 2 });
    expect(stepDown({ tier: "hi", pieces: 2 })).toEqual({ tier: "lite", pieces: 2 });
    expect(stepDown({ tier: "hi", pieces: 1 })).toEqual({ tier: "lite", pieces: 1 });
    expect(stepDown({ tier: "lite", pieces: 3 })).toEqual({ tier: "lite", pieces: 2 });
    expect(stepDown({ tier: "lite", pieces: 2 })).toEqual({ tier: "lite", pieces: 1 });
    expect(stepDown({ tier: "lite", pieces: 1 })).toBeNull();
  });
});

describe("FrameBudget", () => {
  const feed = (b: FrameBudget, ms: number, n: number) => {
    const verdicts = new Set<string>();
    for (let i = 0; i < n; i++) verdicts.add(b.push(ms));
    return verdicts;
  };

  it("skips the warm-up frames, then judges the mean of 90 frames against 20 ms", () => {
    const slow = new FrameBudget({ warmup: 30 });
    // Warm-up frames (shader compiles) are huge and must not count.
    expect(feed(slow, 200, 30)).toEqual(new Set(["sampling"]));
    expect(feed(slow, 21, 89)).toEqual(new Set(["sampling"]));
    expect(slow.push(21)).toBe("slow");

    const fine = new FrameBudget({ warmup: 30 });
    feed(fine, 200, 30);
    feed(fine, 19, 89);
    expect(fine.push(19)).toBe("ok");
  });

  it("uses the mean, so a few long frames do not condemn a fast device", () => {
    const b = new FrameBudget({ warmup: 0 });
    feed(b, 16, 85);
    feed(b, 60, 4);
    expect(b.push(16)).toBe("ok");
  });

  it("ignores gaps longer than a second, as when the tab was hidden", () => {
    const b = new FrameBudget({ warmup: 0 });
    feed(b, 10, 45);
    expect(b.push(5000)).toBe("sampling");
    feed(b, 10, 44);
    expect(b.push(10)).toBe("ok");
  });

  it("starts a fresh round after reset, warm-up included, and says done once judged ok", () => {
    const b = new FrameBudget({ warmup: 10 });
    feed(b, 40, 99);
    expect(b.push(40)).toBe("slow");
    b.reset();
    expect(feed(b, 40, 10)).toEqual(new Set(["sampling"]));
    feed(b, 12, 89);
    expect(b.push(12)).toBe("ok");
    expect(b.push(50)).toBe("done");
  });
});

describe("maxPixelRatio", () => {
  it("allows 2 on hi with a fine pointer, and caps lite or touch devices at 1.5", () => {
    expect(maxPixelRatio({ tier: "hi", pieces: 2 }, false)).toBe(2);
    expect(maxPixelRatio({ tier: "hi", pieces: 3 }, true)).toBe(1.5);
    expect(maxPixelRatio({ tier: "lite", pieces: 2 }, false)).toBe(1.5);
  });
});
