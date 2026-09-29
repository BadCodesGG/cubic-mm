/**
 * The sounds. Everything is synthesised; there are no samples and no audio files.
 *
 * These are factory functions over a plain `BaseAudioContext`, with no module state, so the engine
 * decides when they run and how many may sound at once. A `Voice` is the part that lives in the
 * pool: one PannerNode and one envelope GainNode, built once. Each trigger hangs a few short-lived
 * source nodes off it (an oscillator cannot be restarted, so that is the cheapest thing the Web
 * Audio API allows) and the browser releases them after `stop()`.
 *
 * Space: positions are in µm in the dataset. They are multiplied by `POSITION_SCALE` on the way to
 * the audio graph, so 100 µm is one audio unit, which is the panner's `refDistance`: a source is at
 * full volume inside 100 µm and falls off with `rolloffFactor` 0.6 beyond that.
 */

/** µm to audio units. 100 µm is 1 unit. */
export const POSITION_SCALE = 0.01;
/** Beyond this distance from the camera a spike is not played at all, in µm. */
export const HEAR_RADIUS_UM = 600;

const ATTACK = 0.008;
const DECAY_EXCITATORY = 0.09;
const DECAY_INHIBITORY = 0.14;
const CLICK_DURATION = 0.002;
const WHOOSH_DURATION = 0.3;
/** Slack after an envelope ends before its sources are stopped and its voice counts as free. */
const TAIL = 0.02;
/** How long a stolen voice takes to fade out before its replacement starts. */
export const STEAL_FADE = 0.008;

const BED_GAIN = 0.0316; // -30 dB
const BED_FADE = 1.5;

/** How long each sound occupies a voice, seconds. */
export const DURATION = {
  excitatory: ATTACK + DECAY_EXCITATORY + TAIL,
  inhibitory: ATTACK + DECAY_INHIBITORY + TAIL,
  click: CLICK_DURATION + TAIL,
  whoosh: WHOOSH_DURATION + TAIL,
} as const;

export interface Voice {
  panner: PannerNode;
  env: GainNode;
  /** Sources of the sound currently (or last) on this voice. */
  active: AudioScheduledSourceNode[];
}

export interface Noise {
  white: AudioBuffer;
  brown: AudioBuffer;
}

/** A soft limiter curve: tanh, scaled so full-scale input stays full scale. */
export function tanhCurve(drive = 1.5, points = 1024): Float32Array<ArrayBuffer> {
  const curve = new Float32Array(points);
  const norm = Math.tanh(drive);
  for (let i = 0; i < points; i++) {
    const x = (i / (points - 1)) * 2 - 1;
    curve[i] = Math.tanh(drive * x) / norm;
  }
  return curve;
}

export function createNoise(ctx: BaseAudioContext): Noise {
  const rate = ctx.sampleRate;
  const white = ctx.createBuffer(1, rate, rate);
  const w = white.getChannelData(0);
  for (let i = 0; i < w.length; i++) w[i] = Math.random() * 2 - 1;

  const brown = ctx.createBuffer(1, rate * 3, rate);
  const b = brown.getChannelData(0);
  let last = 0;
  for (let i = 0; i < b.length; i++) {
    last = (last + 0.02 * (Math.random() * 2 - 1)) / 1.02;
    b[i] = last * 3.5;
  }
  // Crossfade the tail into the head so the loop has no seam.
  const fade = Math.floor(rate * 0.25);
  for (let i = 0; i < fade; i++) {
    const k = i / fade;
    b[i] = b[i] * k + b[b.length - fade + i] * (1 - k);
  }
  return { white, brown };
}

export function createVoice(ctx: BaseAudioContext, out: AudioNode): Voice {
  const panner = ctx.createPanner();
  panner.panningModel = "HRTF";
  panner.distanceModel = "inverse";
  panner.refDistance = 1;
  panner.rolloffFactor = 0.6;
  const env = ctx.createGain();
  env.gain.value = 0;
  env.connect(panner).connect(out);
  return { panner, env, active: [] };
}

/** Put a voice at a position given in µm. */
export function placeVoice(voice: Voice, xUm: number, yUm: number, zUm: number): void {
  voice.panner.positionX.value = xUm * POSITION_SCALE;
  voice.panner.positionY.value = yUm * POSITION_SCALE;
  voice.panner.positionZ.value = zUm * POSITION_SCALE;
}

/** Fade a stolen voice out quickly and stop what it was playing. */
export function silence(voice: Voice, now: number): void {
  voice.env.gain.cancelScheduledValues(now);
  voice.env.gain.setTargetAtTime(0, now, STEAL_FADE / 4);
  for (const s of voice.active) {
    try {
      s.stop(now + STEAL_FADE);
    } catch {
      // Already stopped.
    }
  }
  voice.active.length = 0;
}

/** Schedule the amplitude envelope and return the time the sources should stop. */
function shape(voice: Voice, t: number, peak: number, attack: number, decay: number): number {
  const g = voice.env.gain;
  g.cancelScheduledValues(t);
  g.setValueAtTime(0, t);
  g.linearRampToValueAtTime(peak, t + attack);
  g.exponentialRampToValueAtTime(0.0001, t + attack + decay);
  voice.active.length = 0;
  return t + attack + decay + TAIL;
}

function tone(
  ctx: BaseAudioContext,
  voice: Voice,
  type: OscillatorType,
  hz: number,
  level: number,
  t: number,
  end: number,
): void {
  const o = ctx.createOscillator();
  o.type = type;
  o.frequency.value = hz;
  if (level === 1) {
    o.connect(voice.env);
  } else {
    const g = ctx.createGain();
    g.gain.value = level;
    o.connect(g).connect(voice.env);
  }
  o.start(t);
  o.stop(end);
  voice.active.push(o);
}

function noiseSource(ctx: BaseAudioContext, buffer: AudioBuffer, t: number, end: number): AudioBufferSourceNode {
  const s = ctx.createBufferSource();
  s.buffer = buffer;
  s.start(t, Math.random() * (buffer.duration - (end - t)));
  s.stop(end);
  return s;
}

/** Bright, short blip: sine plus a quiet triangle. A stimulated spike is louder and adds the octave. */
export function playExcitatory(ctx: BaseAudioContext, voice: Voice, t: number, hz: number, stimulated: boolean): void {
  const end = shape(voice, t, stimulated ? 0.45 : 0.16, ATTACK, DECAY_EXCITATORY);
  tone(ctx, voice, "sine", hz, 1, t, end);
  tone(ctx, voice, "triangle", hz, 0.3, t, end);
  if (stimulated) tone(ctx, voice, "sine", hz * 2, 0.4, t, end);
}

/** Lower, darker tone with a bandpassed puff of noise on top. */
export function playInhibitory(
  ctx: BaseAudioContext,
  voice: Voice,
  noise: Noise,
  t: number,
  hz: number,
  stimulated: boolean,
): void {
  const end = shape(voice, t, stimulated ? 0.5 : 0.22, ATTACK, DECAY_INHIBITORY);
  tone(ctx, voice, "triangle", hz, 1, t, end);
  if (stimulated) tone(ctx, voice, "sine", hz * 2, 0.4, t, end);
  const band = ctx.createBiquadFilter();
  band.type = "bandpass";
  band.frequency.value = hz * 2.5;
  band.Q.value = 3;
  const level = ctx.createGain();
  level.gain.value = 0.7;
  const n = noiseSource(ctx, noise.white, t, end);
  n.connect(band).connect(level).connect(voice.env);
  voice.active.push(n);
}

/** A 2 ms tick of noise, for a pulse arriving at a synapse. */
export function playClick(ctx: BaseAudioContext, voice: Voice, noise: Noise, t: number): void {
  const end = shape(voice, t, 0.25, 0.0003, CLICK_DURATION);
  const n = noiseSource(ctx, noise.white, t, end);
  n.connect(voice.env);
  voice.active.push(n);
}

/** Filtered noise sweeping up, for a ride jumping to the next cell. Place the voice at the listener. */
export function playWhoosh(ctx: BaseAudioContext, voice: Voice, noise: Noise, t: number): void {
  const end = shape(voice, t, 0.3, 0.08, WHOOSH_DURATION - 0.08);
  const band = ctx.createBiquadFilter();
  band.type = "bandpass";
  band.Q.value = 2;
  band.frequency.setValueAtTime(300, t);
  band.frequency.exponentialRampToValueAtTime(3000, t + WHOOSH_DURATION);
  const n = noiseSource(ctx, noise.white, t, end);
  n.connect(band).connect(voice.env);
  voice.active.push(n);
}

export interface Bed {
  fadeIn(now: number): void;
  /** Fade out, then stop the sources. */
  fadeOut(now: number): void;
}

/**
 * The room tone: two detuned 55 Hz sines and brown noise through a 120 Hz lowpass, at -30 dB.
 * Its nodes exist only between `fadeIn` and the end of `fadeOut`.
 */
export function createBed(ctx: BaseAudioContext, noise: Noise, out: AudioNode): Bed {
  let live: { gain: GainNode; sources: AudioScheduledSourceNode[] } | null = null;
  return {
    fadeIn(now) {
      if (live) {
        live.gain.gain.cancelScheduledValues(now);
        live.gain.gain.setTargetAtTime(BED_GAIN, now, BED_FADE / 3);
        return;
      }
      const gain = ctx.createGain();
      gain.gain.setValueAtTime(0, now);
      gain.gain.setTargetAtTime(BED_GAIN, now, BED_FADE / 3);
      gain.connect(out);
      const sources: AudioScheduledSourceNode[] = [];
      for (const hz of [55, 55.4]) {
        const o = ctx.createOscillator();
        o.frequency.value = hz;
        const g = ctx.createGain();
        g.gain.value = 0.5;
        o.connect(g).connect(gain);
        o.start(now);
        sources.push(o);
      }
      const rumble = ctx.createBufferSource();
      rumble.buffer = noise.brown;
      rumble.loop = true;
      const low = ctx.createBiquadFilter();
      low.type = "lowpass";
      low.frequency.value = 120;
      rumble.connect(low).connect(gain);
      rumble.start(now);
      sources.push(rumble);
      live = { gain, sources };
    },
    fadeOut(now) {
      const current = live;
      if (!current) return;
      live = null;
      current.gain.gain.cancelScheduledValues(now);
      current.gain.gain.setTargetAtTime(0, now, 0.15);
      for (const s of current.sources) s.stop(now + 1);
    },
  };
}
