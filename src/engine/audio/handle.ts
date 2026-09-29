/**
 * Hands the running audio engine from the engine code (which creates it inside `startApp`) to the
 * HUD button (which is React and mounted before the engine exists). A tiny external store, so the
 * button subscribes with `useSyncExternalStore` and neither side imports the other.
 */
import type { AudioEngine } from "./engine";

let current: AudioEngine | null = null;
const listeners = new Set<() => void>();

export function publishAudio(engine: AudioEngine | null): void {
  current = engine;
  for (const l of listeners) l();
}

/** Withdraw `engine`, unless a newer one has already replaced it (dev strict mode disposes late). */
export function unpublishAudio(engine: AudioEngine): void {
  if (current === engine) publishAudio(null);
}

export function subscribeAudio(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function getAudio(): AudioEngine | null {
  return current;
}
