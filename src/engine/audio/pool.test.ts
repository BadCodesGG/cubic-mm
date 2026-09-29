import { describe, expect, it } from "vitest";
import { VoicePool } from "./pool";

describe("VoicePool", () => {
  it("hands out free voices first and reuses one once it has finished", () => {
    const pool = new VoicePool(2, (i) => `v${i}`);
    const a = pool.acquire(0, 1);
    const b = pool.acquire(0.1, 1);
    expect([a.voice, b.voice]).toEqual(["v0", "v1"]);
    expect(a.stolen || b.stolen).toBe(false);
    const c = pool.acquire(1.05, 1); // v0 ended at 1, v1 still sounding
    expect(c).toEqual({ voice: "v0", stolen: false });
  });

  it("steals the oldest sounding voice when all are busy", () => {
    const pool = new VoicePool(32, (i) => i);
    for (let i = 0; i < 32; i++) pool.acquire(i * 0.001, 10);
    const stolen = pool.acquire(0.5, 10);
    expect(stolen).toEqual({ voice: 0, stolen: true });
    // Voice 0 is now the youngest, so the next steal takes voice 1.
    expect(pool.acquire(0.6, 10)).toEqual({ voice: 1, stolen: true });
  });

  it("creates each voice once and never afterwards", () => {
    let created = 0;
    const pool = new VoicePool(4, () => ++created);
    for (let i = 0; i < 100; i++) pool.acquire(i * 0.01, 5);
    expect(created).toBe(4);
  });
});
