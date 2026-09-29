"use client";

import { useCallback, useEffect, useState, type ReactNode } from "react";
import dynamic from "next/dynamic";
import Image from "next/image";
import type { App } from "@/engine/app";
import { readPrefs, writePref } from "@/engine/prefs";

// The HUD reaches into the engine's data helpers (and so into three); it loads with the engine, never
// with the still or the loader.
const Hud = dynamic(() => import("./hud/hud").then((m) => m.Hud), { ssr: false });
import AudioToggle from "./audio-toggle";

interface LoadState {
  fraction: number;
  label: string;
}

/**
 * probing     waiting for idle, then asking the browser what it can draw
 * run         the engine is starting or running
 * still       reduced motion is on: a still frame and an "Enter anyway" button
 * unsupported neither WebGPU nor WebGL 2: the still and an explanation
 */
type Gate = "probing" | "run" | "still" | "unsupported";

/** WebGPU is present, or a WebGL 2 context can be made (three has no WebGL 1 path). */
function canDraw(): boolean {
  if ("gpu" in navigator && navigator.gpu) return true;
  try {
    const gl = document.createElement("canvas").getContext("webgl2");
    // Release the probe's context now rather than waiting on GC; browsers cap live contexts.
    gl?.getExtension("WEBGL_lose_context")?.loseContext();
    return !!gl;
  } catch {
    return false;
  }
}

function whenIdle(fn: () => void): () => void {
  if (typeof window.requestIdleCallback === "function") {
    const id = window.requestIdleCallback(fn, { timeout: 1000 });
    return () => window.cancelIdleCallback(id);
  }
  const id = window.setTimeout(fn, 200);
  return () => window.clearTimeout(id);
}

function Still({ children }: { children?: ReactNode }) {
  return (
    <div className="absolute inset-0">
      <Image
        src="/hero-still.jpg"
        alt="Glowing ribbons of neurons in a cubic millimetre of mouse visual cortex, one axon carrying a bright pulse"
        fill
        priority
        unoptimized
        sizes="100vw"
        className="object-cover"
      />
      <div className="absolute inset-0 bg-gradient-to-t from-[#04060b]/85 via-[#04060b]/5 to-[#04060b]/20" />
      <div className="absolute bottom-8 left-6 max-w-[34rem] pr-6 sm:left-10">
        <h1 className="font-mono text-[11px] uppercase tracking-[0.28em] text-slate-200">One Cubic Millimetre</h1>
        {children}
      </div>
    </div>
  );
}

/**
 * Full-viewport canvas for the brain volume. The engine (and three) is imported on the client
 * only, after mount and after a capability probe at idle, so the server-rendered page is just the
 * dark frame and the loader.
 */
export default function Experience() {
  const [gate, setGate] = useState<Gate>("probing");
  const [load, setLoad] = useState<LoadState>({ fraction: 0, label: "Preparing" });
  const [ready, setReady] = useState(false);
  const [app, setApp] = useState<App | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [canvas, setCanvas] = useState<HTMLCanvasElement | null>(null);

  useEffect(
    () =>
      whenIdle(() => {
        if (!canDraw()) return setGate("unsupported");
        const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        setGate(reduced && readPrefs().motion !== "on" ? "still" : "run");
      }),
    [],
  );

  const enterAnyway = useCallback(() => {
    writePref("motion", "on");
    setGate("run");
  }, []);

  useEffect(() => {
    if (gate !== "run" || !canvas) return;
    let disposed = false;
    let dispose: (() => void) | null = null;
    const params = new URLSearchParams(window.location.search);
    const hold = Number(params.get("t"));
    const prefs = readPrefs();

    import("@/engine/app")
      .then(({ startApp }) =>
        startApp(canvas, {
          shot: params.get("shot"),
          synth: params.get("synth") === "1",
          forceWebGL: params.get("webgl") === "1",
          quality: prefs.quality,
          holdAt: Number.isFinite(hold) && hold > 0 ? hold : undefined,
          select: params.get("select"),
          tour: params.get("tour") === "1", // r3/tour
          onProgress: (fraction, label) => {
            if (!disposed) setLoad({ fraction, label });
          },
        }),
      )
      .then((started) => {
        if (disposed) {
          started.dispose();
          return;
        }
        dispose = started.dispose;
        setApp(started);
        setReady(true);
      })
      .catch((err: unknown) => {
        if (!disposed) setError(err instanceof Error ? err.message : String(err));
      });

    return () => {
      disposed = true;
      dispose?.();
      setApp(null);
      setReady(false);
    };
  }, [gate, canvas]);

  const running = gate === "run";
  const showLoader = gate === "probing" || (running && !ready);

  return (
    <main className="fixed inset-0 overflow-hidden bg-[#04060b] text-slate-300 select-none">
      <canvas
        ref={setCanvas}
        className={`h-full w-full outline-none ${running ? "block" : "hidden"}`}
        aria-label="A cubic millimetre of mouse visual cortex"
      />

      {gate === "still" ? (
        <Still>
          <p className="mt-3 text-[15px] leading-snug text-slate-100">Your device asks for reduced motion, so the volume is paused on a still.</p>
          <p className="mt-1.5 text-[13px] leading-snug text-slate-300">
            Entering starts the live scene: slowly drifting neurons and travelling light.
          </p>
          <button
            type="button"
            onClick={enterAnyway}
            className="mt-4 rounded-[4px] border border-cyan-200/50 bg-cyan-200/10 px-4 py-2 text-[13px] tracking-wide text-cyan-50 transition-colors hover:bg-cyan-200/20 focus-visible:ring-2 focus-visible:ring-cyan-300/80 focus-visible:ring-offset-2 focus-visible:ring-offset-[#04060b] focus-visible:outline-none"
          >
            Enter anyway
          </button>
          <p className="mt-2 text-xs text-slate-300">Remembered on this device.</p>
        </Still>
      ) : null}

      {gate === "unsupported" ? (
        <Still>
          <p className="mt-3 text-[15px] leading-snug text-slate-100">This browser can draw neither WebGPU nor WebGL 2, so the live volume cannot start.</p>
          <p className="mt-1.5 text-[13px] leading-snug text-slate-300">
            Try a recent Chrome, Edge, Firefox or Safari on a device with a graphics chip. The picture is a frame from the live scene.
          </p>
        </Still>
      ) : null}

      <div
        className={`pointer-events-none absolute inset-0 flex items-center justify-center bg-[#04060b] ease-out ${
          showLoader ? "opacity-100" : "opacity-0"
        } ${running ? "transition-opacity duration-[1200ms]" : ""}`}
        aria-hidden={!showLoader}
      >
        <div className="flex w-64 flex-col items-center gap-3">
          <p className="font-mono text-[11px] uppercase tracking-[0.28em] text-slate-400">
            {error ? "Could not start" : load.label}
          </p>
          <div className="h-px w-full overflow-hidden bg-slate-500/20">
            <div
              className="h-full bg-cyan-200/70 transition-[width] duration-300 ease-out"
              style={{ width: `${Math.round(load.fraction * 100)}%` }}
            />
          </div>
          {error ? <p className="text-center text-xs text-rose-300/80">{error}</p> : null}
        </div>
      </div>

      {ready && app ? <Hud app={app} /> : null}
      <AudioToggle />
    </main>
  );
}
