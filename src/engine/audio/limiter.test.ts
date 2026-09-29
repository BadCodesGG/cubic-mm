import { describe, expect, it } from "vitest";
import { RateLimiter } from "./limiter";

describe("RateLimiter", () => {
  it("admits the nearest offers first and drops the rest of the frame", () => {
    const l = new RateLimiter<string>(24, 3);
    l.offer("far", 500);
    l.offer("near", 10);
    l.offer("mid", 200);
    l.offer("nearish", 50);
    expect(l.flush(0)).toEqual(["near", "nearish", "mid"]);
    // "far" was not held over: a spike that is late is a spike that is wrong.
    expect(l.flush(0.016)).toEqual([]);
  });

  it("never admits more than the cap in any one-second window", () => {
    const cap = 24;
    const l = new RateLimiter<number>(cap, 6);
    const admitted: number[] = [];
    for (let frame = 0; frame < 60 * 5; frame++) {
      const now = frame / 60;
      for (let i = 0; i < 50; i++) l.offer(i, i);
      for (let n = l.flush(now).length; n > 0; n--) admitted.push(now);
    }
    expect(admitted.length).toBeGreaterThan(cap * 4);
    for (const start of admitted) {
      const inWindow = admitted.filter((t) => t >= start && t < start + 1).length;
      expect(inWindow).toBeLessThanOrEqual(cap);
    }
  });

  it("recovers its budget once the window has passed", () => {
    const l = new RateLimiter<number>(4, 4);
    for (let i = 0; i < 10; i++) l.offer(i, i);
    expect(l.flush(0)).toHaveLength(4);
    for (let i = 0; i < 10; i++) l.offer(i, i);
    expect(l.flush(0.5)).toHaveLength(0);
    for (let i = 0; i < 10; i++) l.offer(i, i);
    expect(l.flush(1.01)).toHaveLength(4);
  });

  it("stops holding offers once the queue is full", () => {
    const l = new RateLimiter<number>(1000, 1000, 8);
    for (let i = 0; i < 20; i++) l.offer(i, 0);
    expect(l.flush(0)).toHaveLength(8);
  });

  it("can forget pending offers without spending budget", () => {
    const l = new RateLimiter<number>(2, 2);
    l.offer(1, 0);
    l.clear();
    expect(l.flush(0)).toEqual([]);
    l.offer(2, 0);
    l.offer(3, 0);
    expect(l.flush(0.1)).toEqual([2, 3]);
  });
});
