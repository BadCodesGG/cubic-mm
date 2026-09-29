import { describe, expect, it } from "vitest";
import { hash32, pitchHz, EXCITATORY_RANGE_HZ, INHIBITORY_RANGE_HZ } from "./pitch";

describe("hash32", () => {
  it("is the murmur3 finaliser, so a neuron keeps its note across releases", () => {
    // Values computed independently (Python) from the murmur3 fmix32 definition.
    expect([0, 1, 2, 199].map(hash32)).toEqual([0, 1364076727, 821347078, 1485663174]);
  });
});

describe("pitchHz", () => {
  it("gives a cell the same note every time", () => {
    for (let n = 0; n < 50; n++) {
      expect(pitchHz(n, false)).toBe(pitchHz(n, false));
      expect(pitchHz(n, true)).toBe(pitchHz(n, true));
    }
  });

  it("stays inside each class's range and uses more than one note", () => {
    const seen = { exc: new Set<number>(), inh: new Set<number>() };
    for (let n = 0; n < 500; n++) {
      const e = pitchHz(n, false);
      const i = pitchHz(n, true);
      expect(e).toBeGreaterThanOrEqual(EXCITATORY_RANGE_HZ[0]);
      expect(e).toBeLessThanOrEqual(EXCITATORY_RANGE_HZ[1]);
      expect(i).toBeGreaterThanOrEqual(INHIBITORY_RANGE_HZ[0]);
      expect(i).toBeLessThanOrEqual(INHIBITORY_RANGE_HZ[1]);
      seen.exc.add(e);
      seen.inh.add(i);
    }
    expect(seen.exc.size).toBeGreaterThan(3);
    expect(seen.inh.size).toBeGreaterThan(3);
  });

  it("keeps the two classes apart", () => {
    expect(INHIBITORY_RANGE_HZ[1]).toBeLessThan(EXCITATORY_RANGE_HZ[0]);
  });
});
