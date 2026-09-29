/** Uniforms shared by every material in the scene. One object so the look is tuned in one place. */

import { Color } from "three/webgpu";
import { uniform } from "three/tsl";

export function createSceneUniforms() {
  return {
    /** Simulation clock in seconds; spike times are on the same clock. */
    time: uniform(0),
    /** Axonal conduction velocity, m/s. 0.5 is typical for thin unmyelinated cortical axons. */
    velocity: uniform(0.5),
    /**
     * Slow-motion factor applied to conduction delay only. At real speed a spike crosses the
     * whole cube in about 2 ms, far too fast to see travel; 1000x makes it about 2 s.
     */
    slowMo: uniform(1000),
    /** Synaptic delay added on top of the travel time before a pulse lands, seconds. */
    synDelay: uniform(0.001),
    /** Width of the travelling pulse head (a Gaussian in time), seconds of display time. */
    pulseWidth: uniform(0.025),
    /** Decay time of the short tail left behind the pulse head, seconds. */
    afterglow: uniform(0.12),
    spikeColor: uniform(new Color(1.0, 0.6, 0.28)),
    spikeGain: uniform(9),
    afterglowGain: uniform(0.18),
    /** The soma holds its glow longer than the axon, like a calcium transient, but briefly. */
    somaAfterglowGain: uniform(0.25),
    somaAfterglow: uniform(0.4),
    /**
     * View depth at which spike light has halved, µm, on top of the haze. It falls with the fourth
     * power of depth beyond that, so background spikes across the volume stay a faint shimmer and
     * only the few near the visitor read, like distant lightning.
     */
    spikeNearUm: uniform(60),
    /** The same for the stimulated neuron (`stimNeuron`), with a gentler square falloff: its cascade stays bright. */
    cascadeNearUm: uniform(130),
    /** Neuron most recently stimulated (by the visitor or the scripted hero), or -1. Written by app.ts. */
    stimNeuron: uniform(-1),
    /**
     * Largest half-width, device px, of spike light: the glowing core of a thick ribbon and the
     * glow of a soma or a point of light. Up close they stay a bright line or point rather than
     * filling the view with a blurred blob. app.ts scales it with the pixel ratio.
     */
    glowCapPx: uniform(12),
    /** Radius of the point of light riding the pulse front on each axon, µm. */
    pulseSpriteUm: uniform(2.6),
    pulseSpriteGain: uniform(7),
    /** Radius, brightness and decay (s) of the glow on the dendrite where a pulse lands. */
    synapseGlowUm: uniform(1.8),
    synapseGlowGain: uniform(0.8),
    synapseGlowDecay: uniform(0.12),

    /** projectionMatrix[1][1] * drawing-buffer height / 2: multiply by 1/depth for px per µm. */
    pixelScale: uniform(1000),
    /** Ribbons never render thinner than this half-width in device pixels. */
    minHalfWidthPx: uniform(0.8),
    /** Thinnest radius a ribbon is ever given, µm. */
    radiusFloor: uniform(0.12),

    /** View depth over which light falls to 1/e, µm. */
    hazeDistance: uniform(125),
    hazeTint: uniform(new Color(0.07, 0.13, 0.32)),
    exposure: uniform(0.75),
    /** Brightness of ribbons and somas, 1 normally. The intro tour ramps it up from dim (`introFade` in tour.ts). */
    introFade: uniform(1),
    axonGain: uniform(0.7),
    /** Peak opacity of a ribbon; axons are thinner and far more numerous, so they get less. */
    opacity: uniform(0.6),
    axonAlpha: uniform(0.14),
    somaGain: uniform(0.8),

    /** Neuron index the visitor has selected, or -1. Written by app.ts on `select`; materials may read it to highlight. */
    selectedNeuron: uniform(-1),
  };
}

export type SceneUniforms = ReturnType<typeof createSceneUniforms>;
