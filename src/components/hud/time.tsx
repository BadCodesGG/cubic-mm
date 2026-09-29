"use client";

import type { App } from "@/engine/app";
import { FOCUS } from "./panels";
import { useTimeScale } from "./use-hud";

const SPEEDS = [
  { scale: 0, label: "Pause", title: "Pause the simulation" },
  { scale: 0.1, label: "0.1x", title: "Slow motion, a tenth of the speed ( , )" },
  { scale: 1, label: "1x", title: "Normal speed ( . )" },
] as const;

/** Pause, 0.1x and 1x. The simulation and the ride slow with it; flying the camera does not. */
export function TimeControl({ app }: { app: App }) {
  const scale = useTimeScale(app);
  return (
    <div role="group" aria-label="Simulation speed" className="pointer-events-auto mt-2 inline-flex gap-1">
      {SPEEDS.map((s) => (
        <button
          key={s.scale}
          type="button"
          title={s.title}
          aria-pressed={scale === s.scale}
          onClick={() => app.bus.emit("timeScale", { scale: s.scale })}
          className={`h-7 min-w-9 rounded-[3px] border px-2 font-mono text-[11px] tracking-wide backdrop-blur-sm transition-colors ${
            scale === s.scale
              ? "border-cyan-200/50 bg-cyan-200/15 text-cyan-50"
              : "border-slate-500/30 bg-slate-900/40 text-slate-300 hover:border-slate-400/50 hover:bg-slate-800/50"
          } ${FOCUS}`}
        >
          {s.label}
        </button>
      ))}
    </div>
  );
}

/** "PAUSED" or "0.1x" under the status line; empty at normal speed (the live region stays mounted so changes are announced). */
export function TimeStatus({ app }: { app: App }) {
  const scale = useTimeScale(app);
  return (
    <p role="status" className="mt-1 font-mono text-[11px] tracking-[0.16em] text-amber-200 empty:hidden">
      {scale === 1 ? "" : scale === 0 ? "PAUSED" : `${scale}x`}
    </p>
  );
}
