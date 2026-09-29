/**
 * Rate discipline for a stream of sounds. Offers arrive whenever the bus emits; once per frame the
 * caller flushes, and the limiter admits the nearest ones that fit a sliding one-second budget.
 * Whatever does not fit is dropped rather than carried over, because a late spike is a wrong spike.
 */
export class RateLimiter<T> {
  private items: T[] = [];
  private distances: number[] = [];
  /** Admission times inside the current window, oldest first. */
  private stamps: number[] = [];

  /**
   * @param perSecond Hard cap on admissions in any one-second window.
   * @param perFlush Cap on one flush, so a burst does not spend the whole second's budget on one frame.
   * @param maxPending Offers beyond this between two flushes are ignored.
   */
  constructor(
    private readonly perSecond: number,
    private readonly perFlush: number = perSecond,
    private readonly maxPending: number = 256,
  ) {}

  offer(item: T, distance: number): void {
    if (this.items.length >= this.maxPending) return;
    this.items.push(item);
    this.distances.push(distance);
  }

  /** Forget everything offered since the last flush. */
  clear(): void {
    this.items = [];
    this.distances = [];
  }

  /** The admitted offers, nearest first. Clears everything else. */
  flush(now: number): T[] {
    let drop = 0;
    while (drop < this.stamps.length && this.stamps[drop] <= now - 1) drop++;
    if (drop) this.stamps.splice(0, drop);

    const room = Math.min(this.perFlush, this.perSecond - this.stamps.length);
    const order = this.items.map((_, i) => i).sort((a, b) => this.distances[a] - this.distances[b]);
    const admitted = order.slice(0, Math.max(0, room)).map((i) => this.items[i]);
    for (let i = 0; i < admitted.length; i++) this.stamps.push(now);

    this.items = [];
    this.distances = [];
    return admitted;
  }
}
