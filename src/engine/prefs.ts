/**
 * The visitor's saved choices in localStorage, each validated on read: anything unexpected (a
 * stale value, a hand edit, storage that throws in a privacy mode) falls back to the default.
 */

export type Quality = "auto" | "hi" | "lite";
export type Motion = "auto" | "on";
export type Audio = "on" | "off";

export interface Prefs {
  /**
   * "hi" and "lite" pin the render level (see quality.ts); "auto" picks one from the device and
   * steps it down while frames run slow. The backend (WebGPU or WebGL2) is chosen separately.
   */
  quality: Quality;
  /** "on" plays the volume even when the OS asks for reduced motion. */
  motion: Motion;
  audio: Audio;
}

export const PREF_KEYS = { quality: "cmm-quality", motion: "cmm-motion", audio: "cmm-audio" } as const;

const CHOICES: { [K in keyof Prefs]: readonly Prefs[K][] } = {
  quality: ["auto", "hi", "lite"],
  motion: ["auto", "on"],
  audio: ["on", "off"],
};

export const DEFAULT_PREFS: Readonly<Prefs> = { quality: "auto", motion: "auto", audio: "on" };

type Reader = Pick<Storage, "getItem">;
type Writer = Pick<Storage, "setItem">;

function defaultStorage(): Storage | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage;
  } catch {
    return null;
  }
}

export function readPref<K extends keyof Prefs>(key: K, storage: Reader | null = defaultStorage()): Prefs[K] {
  let raw: string | null = null;
  try {
    raw = storage?.getItem(PREF_KEYS[key]) ?? null;
  } catch {
    // A failed read is no value.
  }
  const choices = CHOICES[key] as readonly string[];
  return raw !== null && choices.includes(raw) ? (raw as Prefs[K]) : DEFAULT_PREFS[key];
}

export function readPrefs(storage: Reader | null = defaultStorage()): Prefs {
  return { quality: readPref("quality", storage), motion: readPref("motion", storage), audio: readPref("audio", storage) };
}

/** Persists a choice; returns false when the value is invalid or storage refused it. */
export function writePref<K extends keyof Prefs>(key: K, value: Prefs[K], storage: Writer | null = defaultStorage()): boolean {
  if (!(CHOICES[key] as readonly string[]).includes(value)) return false;
  try {
    if (!storage) return false;
    storage.setItem(PREF_KEYS[key], value);
    return true;
  } catch {
    // Not persisted this visit; the choice still applies.
    return false;
  }
}
