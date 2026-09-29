"use client";

import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import type { App } from "@/engine/app";
import type { RideState } from "@/engine/camera/ride";

/** The selected neuron index, -1 for none. */
export function useSelection(app: App): number {
  return useSyncExternalStore(
    (notify) => app.bus.on("select", notify),
    () => app.selection(),
    () => -1,
  );
}

function subscribeMedia(query: string) {
  return (notify: () => void) => {
    const m = window.matchMedia(query);
    m.addEventListener("change", notify);
    return () => m.removeEventListener("change", notify);
  };
}

/** Live `matchMedia` result; `serverValue` while rendering on the server. */
export function useMedia(query: string, serverValue = false): boolean {
  return useSyncExternalStore(
    subscribeMedia(query),
    () => window.matchMedia(query).matches,
    () => serverValue,
  );
}

export const useReducedMotion = () => useMedia("(prefers-reduced-motion: reduce)");
export const useCoarsePointer = () => useMedia("(pointer: coarse)");

export interface FrameSnap {
  scaleUm: number;
  scalePx: number;
  /** Degrees, 0 is up, clockwise. */
  piaDeg: number;
  /** 0..1 */
  piaLength: number;
  ring: [number, number] | null;
  ride: RideState;
  rideNeuron: number;
  locked: boolean;
}

function snapshot(app: App): FrameSnap {
  const h = app.hud;
  return {
    scaleUm: h.scaleUm,
    scalePx: Math.round(h.scalePx),
    piaDeg: Math.round((h.piaAngle * 180) / Math.PI),
    piaLength: Math.round(h.piaLength * 20) / 20,
    ring: h.selectedScreen ? [Math.round(h.selectedScreen[0]), Math.round(h.selectedScreen[1])] : null,
    ride: h.ride,
    rideNeuron: h.rideNeuron,
    locked: h.locked,
  };
}

const same = (a: FrameSnap, b: FrameSnap) =>
  a.scaleUm === b.scaleUm &&
  a.scalePx === b.scalePx &&
  a.piaDeg === b.piaDeg &&
  a.piaLength === b.piaLength &&
  a.ride === b.ride &&
  a.rideNeuron === b.rideNeuron &&
  a.locked === b.locked &&
  a.ring?.[0] === b.ring?.[0] &&
  a.ring?.[1] === b.ring?.[1];

/** Polls the engine's per-frame HUD numbers each animation frame, re-rendering only when a rounded value changed. */
export function useHudFrame(app: App): FrameSnap {
  const [snap, setSnap] = useState(() => snapshot(app));
  const last = useRef(snap);
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const next = snapshot(app);
      if (!same(next, last.current)) {
        last.current = next;
        setSnap(next);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [app]);
  return snap;
}

export interface Status {
  gpu: boolean;
  neurons: number;
  synapses: number | null;
  frameMs: number;
  /** Render tier now drawn ("hi" or "lite"); it can drop while running on auto. */
  tier: string | null;
}

/** Neuron and synapse counts and backend from the `mode` event, frame time and tier from `window.__cmm` twice a second. */
export function useStatus(app: App): Status {
  const [status, setStatus] = useState<Status>(() => ({
    gpu: window.__cmm?.sim?.mode === "gpu",
    neurons: app.data.neurons.count,
    synapses: app.data.synapses ? app.data.synapses.count : null,
    frameMs: window.__cmm?.frameMs ?? 0,
    tier: window.__cmm?.quality.tier ?? null,
  }));
  useEffect(() => {
    const off = app.bus.on("mode", (m) =>
      setStatus((s) => ({ ...s, gpu: m.gpu, neurons: m.neuronCount, synapses: m.synapseCount > 0 ? m.synapseCount : null })),
    );
    const id = window.setInterval(() => {
      const ms = window.__cmm?.frameMs ?? 0;
      const tier = window.__cmm?.quality.tier ?? null;
      setStatus((s) => (Math.abs(s.frameMs - ms) < 0.05 && s.tier === tier ? s : { ...s, frameMs: ms, tier }));
    }, 500);
    return () => {
      off();
      window.clearInterval(id);
    };
  }, [app]);
  return status;
}
