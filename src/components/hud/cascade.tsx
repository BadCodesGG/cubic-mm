"use client";

import { useEffect, useRef, useState } from "react";
import type { App } from "@/engine/app";
import { TIMELINE_S, type CascadeSummary, type CascadeTick } from "@/engine/cascade";
import { FOCUS } from "./panels";

interface CascadeView {
  summary: CascadeSummary;
  ticks: CascadeTick[];
  elapsedS: number;
  latest: number;
}

const POLL_MS = 100;

/** The active cascade, or null once it is stale. Polled, and only re-rendered when a shown value changed. */
function useCascade(app: App): CascadeView | null {
  const [view, setView] = useState<CascadeView | null>(null);
  const key = useRef("");
  useEffect(() => {
    const read = () => {
      const t = app.cascade;
      if (!t.active) {
        if (key.current !== "") {
          key.current = "";
          setView(null);
        }
        return;
      }
      const summary = t.summary();
      const ticks = t.timeline();
      const elapsedS = Math.round(t.elapsedS() * 10) / 10;
      const next = `${summary.root}:${summary.reached}:${summary.hop1}:${summary.hop2}:${ticks.length}:${t.latest()}:${elapsedS}`;
      if (next === key.current) return;
      key.current = next;
      setView({ summary, ticks, elapsedS, latest: t.latest() });
    };
    const id = window.setInterval(read, POLL_MS);
    read();
    return () => window.clearInterval(id);
  }, [app]);
  return view;
}

const plural = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;

function sentence(s: CascadeSummary): string {
  const reached = `reached ${plural(s.reached, "cell", "cells")}`;
  if (s.hop1 === 0) return `${reached}, none has fired yet`;
  const fired = `${s.hop1} fired within ${s.hop1SpanS.toFixed(1)} s`;
  const more = s.hop2 > 0 ? `, at least ${s.hop2} more one hop later` : "";
  return `${reached}, ${fired}${more}`;
}

const STRIP_W = 268;
const STRIP_H = 18;

/** Eight seconds, one tick per spike the cascade caused. After 8 s the window slides with the clock. */
function Timeline({ ticks, elapsedS }: { ticks: CascadeTick[]; elapsedS: number }) {
  const start = Math.max(0, elapsedS - TIMELINE_S);
  const x = (t: number) => Math.min(STRIP_W - 1, Math.max(1, ((t - start) / TIMELINE_S) * STRIP_W));
  return (
    <svg
      role="img"
      aria-label={`Timeline of the last ${TIMELINE_S} seconds: ${ticks.length} spikes caused`}
      viewBox={`0 0 ${STRIP_W} ${STRIP_H}`}
      className="mt-2 block h-[18px] w-full max-w-full"
      preserveAspectRatio="none"
    >
      <line x1="0" x2={STRIP_W} y1={STRIP_H - 0.5} y2={STRIP_H - 0.5} stroke="currentColor" className="text-slate-500/50" />
      {start === 0 ? <line x1="0.5" x2="0.5" y1="0" y2={STRIP_H} stroke="currentColor" className="text-cyan-200/70" /> : null}
      {ticks.map((k) => (
        <line
          key={`${k.neuron}:${k.t}`}
          x1={x(k.t)}
          x2={x(k.t)}
          y1={k.hop === 1 ? 2 : 7}
          y2={STRIP_H - 2}
          stroke="currentColor"
          strokeWidth="1.5"
          className={k.hop === 1 ? "text-amber-200" : "text-amber-500/80"}
        />
      ))}
    </svg>
  );
}

/** Under the cell panel while a cascade is active: what the last stimulus set off, and a way to chase it. */
export function CascadePanel({ app, selected }: { app: App; selected: number }) {
  const view = useCascade(app);
  if (!view) return null;
  const { summary, ticks, elapsedS, latest } = view;
  const from = selected === summary.root ? "this cell" : "the cell you stimulated";
  return (
    <section aria-label="Cascade" className="mt-2.5 w-[19.5rem] max-w-[calc(100vw-2rem)] border-t border-slate-500/25 pt-2.5">
      <p className="text-[12px] leading-snug text-slate-300">
        <span className="font-mono text-[11px] uppercase tracking-[0.12em] text-amber-200">Cascade from {from}</span>
        <span className="mt-0.5 block text-slate-200">{sentence(summary)}.</span>
      </p>
      <Timeline ticks={ticks} elapsedS={elapsedS} />
      <div className="mt-0.5 flex justify-between font-mono text-[11px] text-slate-400">
        <span>{elapsedS > TIMELINE_S ? `${(elapsedS - TIMELINE_S).toFixed(0)} s` : "stimulus"}</span>
        <span>{elapsedS > TIMELINE_S ? `${elapsedS.toFixed(0)} s` : `${TIMELINE_S} s`}</span>
      </div>
      <button
        type="button"
        disabled={latest < 0}
        onClick={() => app.bus.emit("select", { neuron: latest })}
        aria-label="Follow: select the cell that fired most recently"
        title="Select the cell that fired most recently"
        className={`mt-2 inline-flex items-center rounded-[4px] border border-slate-500/30 bg-slate-900/40 px-2.5 py-1.5 text-[11px] tracking-wide text-slate-300 transition-colors hover:border-slate-400/50 hover:bg-slate-800/50 disabled:cursor-not-allowed disabled:opacity-50 pointer-coarse:px-3 pointer-coarse:py-2.5 ${FOCUS}`}
      >
        Follow
      </button>
    </section>
  );
}
