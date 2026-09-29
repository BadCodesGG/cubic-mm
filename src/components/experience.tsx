"use client";

import { useEffect, useRef, useState } from "react";
import AudioToggle from "./audio-toggle";

interface LoadState {
  fraction: number;
  label: string;
}

/**
 * Full-viewport canvas for the brain volume. The engine (and three) is imported on the client
 * only, after mount, so the server-rendered page is just the dark frame and the loader.
 */
export default function Experience() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [load, setLoad] = useState<LoadState>({ fraction: 0, label: "Preparing" });
  const [ready, setReady] = useState(false);
  const [lite, setLite] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let disposed = false;
    let dispose: (() => void) | null = null;
    const params = new URLSearchParams(window.location.search);
    const hold = Number(params.get("t"));

    import("@/engine/app")
      .then(({ startApp }) =>
        startApp(canvas, {
          shot: params.get("shot"),
          synth: params.get("synth") === "1",
          forceWebGL: params.get("webgl") === "1",
          holdAt: Number.isFinite(hold) && hold > 0 ? hold : undefined,
          onProgress: (fraction, label) => {
            if (!disposed) setLoad({ fraction, label });
          },
        }),
      )
      .then((app) => {
        if (disposed) {
          app.dispose();
          return;
        }
        dispose = app.dispose;
        setLite(!app.isWebGPU);
        setReady(true);
      })
      .catch((err: unknown) => {
        if (!disposed) setError(err instanceof Error ? err.message : String(err));
      });

    return () => {
      disposed = true;
      dispose?.();
    };
  }, []);

  return (
    <main className="fixed inset-0 overflow-hidden bg-[#04060b] text-slate-300 select-none">
      <canvas ref={canvasRef} className="block h-full w-full outline-none" aria-label="A cubic millimetre of mouse visual cortex" />

      <div
        className={`pointer-events-none absolute inset-0 flex items-center justify-center bg-[#04060b] transition-opacity duration-[1200ms] ease-out ${
          ready ? "opacity-0" : "opacity-100"
        }`}
        aria-hidden={ready}
      >
        <div className="flex w-64 flex-col items-center gap-3">
          <p className="font-mono text-[11px] uppercase tracking-[0.28em] text-slate-400/80">
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

      {lite ? (
        <div className="pointer-events-none absolute right-4 top-4 rounded-full border border-slate-400/20 bg-slate-900/40 px-3 py-1 font-mono text-[10px] uppercase tracking-[0.2em] text-slate-400 backdrop-blur-sm">
          WebGL2 lite mode
        </div>
      ) : null}

      <AudioToggle />

      <p className="pointer-events-none absolute bottom-5 left-6 max-w-[80vw] text-[12px] tracking-wide text-slate-400/75">
        <span className="text-slate-200/90">One Cubic Millimetre</span>
        <span className="mx-2 text-slate-500">·</span>
        200 real neurons from the MICrONS mouse visual cortex
      </p>
    </main>
  );
}
