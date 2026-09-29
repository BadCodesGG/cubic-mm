import { describe, expect, it } from "vitest";
import { AUDIO_PREF_KEY, readAudioPref, writeAudioPref, type PrefStorage } from "./pref";

function memory(initial: Record<string, string> = {}): PrefStorage & { data: Record<string, string> } {
  const data = { ...initial };
  return {
    data,
    getItem: (k) => data[k] ?? null,
    setItem: (k, v) => {
      data[k] = v;
    },
  };
}

const broken: PrefStorage = {
  getItem: () => {
    throw new Error("denied");
  },
  setItem: () => {
    throw new Error("denied");
  },
};

describe("audio preference", () => {
  it("uses the shared cmm-audio key", () => {
    expect(AUDIO_PREF_KEY).toBe("cmm-audio");
  });

  it("round-trips on and off", () => {
    const s = memory();
    expect(readAudioPref(s)).toBeNull();
    writeAudioPref("on", s);
    expect(s.data["cmm-audio"]).toBe("on");
    expect(readAudioPref(s)).toBe("on");
    writeAudioPref("off", s);
    expect(readAudioPref(s)).toBe("off");
  });

  it("treats anything else as unset", () => {
    expect(readAudioPref(memory({ "cmm-audio": "yes" }))).toBeNull();
  });

  it("survives storage that throws or is missing", () => {
    expect(readAudioPref(broken)).toBeNull();
    expect(() => writeAudioPref("on", broken)).not.toThrow();
    expect(readAudioPref(undefined)).toBeNull();
  });
});
