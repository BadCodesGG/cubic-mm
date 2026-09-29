/**
 * Spatial audio for the volume: every spike is a short synthesised sound at its soma, heard from
 * wherever the camera is. No samples, nothing fetched.
 *
 * Graph:
 *
 *   voice (osc / noise -> envelope GainNode -> HRTF PannerNode) x 32  --+
 *   bed (2 sines + brown noise -> lowpass -> GainNode, -30 dB)  --------+
 *                                                                       v
 *   DynamicsCompressorNode -> WaveShaperNode (tanh soft limiter) -> master GainNode (-6 dB) -> destination
 *
 * The AudioContext is created on the first `enable()`, which must come from a user gesture. Until
 * then, and after `disable()`, nothing is scheduled: bus events are ignored and `update` returns.
 * Positions are in µm; see `POSITION_SCALE` in `voices.ts`.
 */

import type { EventBus } from "../events";
import type { Dataset } from "../data";
import { dequantise } from "../format";
import { RateLimiter } from "./limiter";
import { pitchHz } from "./pitch";
import { VoicePool } from "./pool";
import {
  DURATION,
  HEAR_RADIUS_UM,
  POSITION_SCALE,
  STEAL_FADE,
  createBed,
  createNoise,
  createVoice,
  placeVoice,
  playClick,
  playExcitatory,
  playInhibitory,
  playWhoosh,
  silence,
  tanhCurve,
  type Bed,
  type Noise,
  type Voice,
} from "./voices";

/** The slice of a three.js camera the listener needs; keeps this module free of three. */
export interface AudioCamera {
  matrixWorld: { elements: ArrayLike<number> };
  updateMatrixWorld(force?: boolean): void;
}

export interface AudioOptions {
  bus: EventBus;
  dataset: Dataset;
  getCamera: () => AudioCamera;
}

export interface AudioEngine {
  readonly enabled: boolean;
  /** Start (or resume) sound. Call from a click. Resolves true once the context is running. */
  enable(): Promise<boolean>;
  disable(): void;
  /** Once per frame: move the listener and play what the limiters admit. */
  update(dt: number): void;
  dispose(): void;
}

const VOICES = 32;
const SPIKES_PER_SECOND = 24;
const SPIKES_PER_FRAME = 6;
const CLICKS_PER_SECOND = 12;
const CLICKS_PER_FRAME = 3;
const MASTER_GAIN = 0.501; // -6 dB
const SUSPEND_AFTER_MS = 1500;

interface SpikeItem {
  neuron: number;
  stimulated: boolean;
}

interface ClickItem {
  x: number;
  y: number;
  z: number;
}

export function createAudio({ bus, dataset, getCamera }: AudioOptions): AudioEngine {
  const { somaUm, inhibitory } = dataset.neurons;
  const { min, max } = dataset.manifest.boundsUm;
  const synapses = dataset.synapses;

  let ctx: AudioContext | null = null;
  let compressor: DynamicsCompressorNode;
  let noise: Noise;
  let bed: Bed;
  let pool: VoicePool<Voice>;
  let enabled = false;
  let disposed = false;
  let suspendTimer: ReturnType<typeof setTimeout> | undefined;

  const spikes = new RateLimiter<SpikeItem>(SPIKES_PER_SECOND, SPIKES_PER_FRAME);
  const clicks = new RateLimiter<ClickItem>(CLICKS_PER_SECOND, CLICKS_PER_FRAME);

  // Listener position in µm, refreshed each `update`; the bus handlers measure distance against it.
  let lx = 0;
  let ly = 0;
  let lz = 0;

  const distanceUm = (x: number, y: number, z: number) => Math.hypot(x - lx, y - ly, z - lz);

  function build(): AudioContext {
    const c = new AudioContext({ latencyHint: "interactive" });
    compressor = c.createDynamicsCompressor();
    compressor.threshold.value = -18;
    compressor.knee.value = 12;
    compressor.ratio.value = 4;
    compressor.attack.value = 0.003;
    compressor.release.value = 0.2;
    const limiter = c.createWaveShaper();
    limiter.curve = tanhCurve();
    limiter.oversample = "2x";
    const master = c.createGain();
    master.gain.value = MASTER_GAIN;
    compressor.connect(limiter).connect(master).connect(c.destination);

    noise = createNoise(c);
    bed = createBed(c, noise, compressor);
    pool = new VoicePool(VOICES, () => createVoice(c, compressor));
    return c;
  }

  /**
   * Claim a voice and return it with its start time. A stolen voice is faded out first, so its
   * replacement starts STEAL_FADE later; the pool's bookkeeping for it runs 8 ms short, which only
   * ever overlaps the tail of a sound that is already fading.
   */
  function claim(c: AudioContext, duration: number): { voice: Voice; t: number } {
    const now = c.currentTime;
    const { voice, stolen } = pool.acquire(now, duration);
    if (stolen) silence(voice, now);
    return { voice, t: stolen ? now + STEAL_FADE : now };
  }

  function playSpike(c: AudioContext, item: SpikeItem): void {
    const inh = inhibitory[item.neuron] === 1;
    const { voice, t } = claim(c, inh ? DURATION.inhibitory : DURATION.excitatory);
    placeVoice(voice, somaUm[item.neuron * 3], somaUm[item.neuron * 3 + 1], somaUm[item.neuron * 3 + 2]);
    const hz = pitchHz(item.neuron, inh);
    if (inh) playInhibitory(c, voice, noise, t, hz, item.stimulated);
    else playExcitatory(c, voice, t, hz, item.stimulated);
  }

  function playTick(c: AudioContext, item: ClickItem): void {
    const { voice, t } = claim(c, DURATION.click);
    placeVoice(voice, item.x, item.y, item.z);
    playClick(c, voice, noise, t);
  }

  function updateListener(c: AudioContext): void {
    const cam = getCamera();
    cam.updateMatrixWorld();
    const e = cam.matrixWorld.elements;
    // Column 3 is the position; column 1 is up; column 2 points backwards, so forward is its negation.
    lx = e[12];
    ly = e[13];
    lz = e[14];
    const l = c.listener;
    const now = c.currentTime;
    if (l.positionX) {
      l.positionX.setValueAtTime(lx * POSITION_SCALE, now);
      l.positionY.setValueAtTime(ly * POSITION_SCALE, now);
      l.positionZ.setValueAtTime(lz * POSITION_SCALE, now);
      l.forwardX.setValueAtTime(-e[8], now);
      l.forwardY.setValueAtTime(-e[9], now);
      l.forwardZ.setValueAtTime(-e[10], now);
      l.upX.setValueAtTime(e[4], now);
      l.upY.setValueAtTime(e[5], now);
      l.upZ.setValueAtTime(e[6], now);
    } else {
      l.setPosition(lx * POSITION_SCALE, ly * POSITION_SCALE, lz * POSITION_SCALE);
      l.setOrientation(-e[8], -e[9], -e[10], e[4], e[5], e[6]);
    }
  }

  const offs = [
    bus.on("spike", ({ neuron, stimulated }) => {
      if (!enabled) return;
      const d = distanceUm(somaUm[neuron * 3], somaUm[neuron * 3 + 1], somaUm[neuron * 3 + 2]);
      if (d <= HEAR_RADIUS_UM) spikes.offer({ neuron, stimulated }, d);
    }),
    bus.on("arrive", ({ post, synapse }) => {
      if (!enabled) return;
      // Without a synapse table (or an index into it) the click falls back to the target's soma.
      let x = somaUm[post * 3];
      let y = somaUm[post * 3 + 1];
      let z = somaUm[post * 3 + 2];
      if (synapses && synapse >= 0 && synapse < synapses.count) {
        x = dequantise(synapses.pos[synapse * 3], min[0], max[0]);
        y = dequantise(synapses.pos[synapse * 3 + 1], min[1], max[1]);
        z = dequantise(synapses.pos[synapse * 3 + 2], min[2], max[2]);
      }
      const d = distanceUm(x, y, z);
      if (d <= HEAR_RADIUS_UM) clicks.offer({ x, y, z }, d);
    }),
    bus.on("rideJump", () => {
      if (!enabled || !ctx) return;
      const { voice, t } = claim(ctx, DURATION.whoosh);
      placeVoice(voice, lx, ly, lz);
      playWhoosh(ctx, voice, noise, t);
    }),
  ];

  return {
    get enabled() {
      return enabled;
    },

    async enable() {
      if (disposed) return false;
      clearTimeout(suspendTimer);
      ctx ??= build();
      const c = ctx;
      // `resume()` can stay pending when the browser has not seen a gesture, so do not wait on it forever.
      await Promise.race([c.resume(), new Promise((r) => setTimeout(r, 500))]).catch(() => {});
      if (disposed) return false;
      if (c.state !== "running") {
        void c.suspend().catch(() => {});
        return false;
      }
      enabled = true;
      updateListener(c);
      bed.fadeIn(c.currentTime);
      return true;
    },

    disable() {
      if (!enabled || !ctx) return;
      enabled = false;
      // Drop what was queued for the next frame; a sound that arrives after "off" would be a bug.
      spikes.clear();
      clicks.clear();
      const c = ctx;
      bed.fadeOut(c.currentTime);
      suspendTimer = setTimeout(() => {
        if (!enabled) void c.suspend().catch(() => {});
      }, SUSPEND_AFTER_MS);
    },

    update() {
      if (!enabled || !ctx) return;
      updateListener(ctx);
      for (const item of spikes.flush(ctx.currentTime)) playSpike(ctx, item);
      for (const item of clicks.flush(ctx.currentTime)) playTick(ctx, item);
    },

    dispose() {
      disposed = true;
      enabled = false;
      clearTimeout(suspendTimer);
      for (const off of offs) off();
      void ctx?.close().catch(() => {});
      ctx = null;
    },
  };
}
