/**
 * Which note a neuron sings. A hash of the neuron index picks one step of a small scale, so a
 * cell always has the same pitch and the sound of a spike identifies the cell.
 */

/** Excitatory cells sing high and bright; inhibitory cells low and dark. Ranges are inclusive, in Hz. */
export const EXCITATORY_RANGE_HZ = [900, 1600] as const;
export const INHIBITORY_RANGE_HZ = [280, 480] as const;

/** Just-intonation ratios above the root; the last one lands exactly on the top of the range. */
const EXCITATORY_RATIOS = [1, 9 / 8, 5 / 4, 3 / 2, 5 / 3, 16 / 9] as const;
const INHIBITORY_RATIOS = [1, 9 / 8, 5 / 4, 4 / 3, 3 / 2, 12 / 7] as const;

/** murmur3 32-bit finaliser: a stateless, well-mixed hash of a small integer. */
export function hash32(n: number): number {
  let h = n >>> 0;
  h ^= h >>> 16;
  h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return h >>> 0;
}

export function pitchHz(neuron: number, inhibitory: boolean): number {
  const [root] = inhibitory ? INHIBITORY_RANGE_HZ : EXCITATORY_RANGE_HZ;
  const ratios = inhibitory ? INHIBITORY_RATIOS : EXCITATORY_RATIOS;
  return root * ratios[hash32(neuron) % ratios.length];
}
