/**
 * The visitor's sound choice, persisted under `cmm-audio`. The prefs module elsewhere in the app
 * adopts the same key, so this reads and writes storage directly and never trusts what it finds.
 */
export const AUDIO_PREF_KEY = "cmm-audio";

export type AudioPref = "on" | "off";

export interface PrefStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

/** localStorage throws in some privacy modes and does not exist on the server. */
function defaultStorage(): PrefStorage | undefined {
  try {
    return globalThis.localStorage;
  } catch {
    return undefined;
  }
}

export function readAudioPref(storage: PrefStorage | undefined = defaultStorage()): AudioPref | null {
  try {
    const v = storage?.getItem(AUDIO_PREF_KEY);
    return v === "on" || v === "off" ? v : null;
  } catch {
    return null;
  }
}

export function writeAudioPref(value: AudioPref, storage: PrefStorage | undefined = defaultStorage()): void {
  try {
    storage?.setItem(AUDIO_PREF_KEY, value);
  } catch {
    // Not persisting is fine; the choice still holds for this visit.
  }
}
