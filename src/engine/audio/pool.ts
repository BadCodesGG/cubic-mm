/**
 * A fixed set of reusable voices. Every voice is built once up front; a new sound takes a free one,
 * or steals the one that started longest ago when none is free, so the number of live audio nodes
 * is bounded no matter how many spikes arrive.
 */
interface Slot<T> {
  voice: T;
  start: number;
  /** Audio-clock time at which the voice falls silent. */
  until: number;
}

export class VoicePool<T> {
  private readonly slots: Slot<T>[];

  constructor(size: number, create: (index: number) => T) {
    this.slots = Array.from({ length: size }, (_, i) => ({ voice: create(i), start: -Infinity, until: -Infinity }));
  }

  /** Claim a voice for a sound lasting `duration` seconds from `now`. */
  acquire(now: number, duration: number): { voice: T; stolen: boolean } {
    let pick = this.slots.find((s) => s.until <= now);
    const stolen = !pick;
    if (!pick) pick = this.slots.reduce((oldest, s) => (s.start < oldest.start ? s : oldest));
    pick.start = now;
    pick.until = now + duration;
    return { voice: pick.voice, stolen };
  }
}
