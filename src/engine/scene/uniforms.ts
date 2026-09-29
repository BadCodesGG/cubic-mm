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
    /** Width of the travelling pulse, seconds of display time. */
    pulseWidth: uniform(0.06),
    /** Decay time of the afterglow left behind the pulse, seconds. */
    afterglow: uniform(0.25),
    spikeColor: uniform(new Color(1.0, 0.6, 0.28)),
    spikeGain: uniform(9),
    afterglowGain: uniform(0.35),
    /** The soma holds its glow longer and brighter than the axon, like a calcium transient. */
    somaAfterglowGain: uniform(0.6),
    somaAfterglow: uniform(0.9),

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
