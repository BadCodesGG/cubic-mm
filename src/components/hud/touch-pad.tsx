"use client";

import { useEffect, useRef, type PointerEvent } from "react";
import type { App } from "@/engine/app";
import { NO_STICK, stickVector, type Stick } from "@/engine/camera/touch";

const RADIUS = 52;

/**
 * The phone's dual sticks: the left one moves, the right one looks (`engine/camera/touch.ts` turns a
 * finger offset into a stick vector; `FlyControls.setSticks` applies it). Each stick tracks its own
 * pointer, so both thumbs work at once.
 */
export function TouchPad({ app }: { app: App }) {
  const move = useRef<Stick>(NO_STICK);
  const look = useRef<Stick>(NO_STICK);

  useEffect(
    () => () => {
      app.setSticks(NO_STICK, NO_STICK);
    },
    [app],
  );

  const push = () => app.setSticks(move.current, look.current);

  return (
    <div className="pointer-events-none absolute inset-x-0 bottom-0 flex items-end justify-between px-6 pb-8">
      <StickZone label="Move" onChange={(s) => { move.current = s; push(); }} />
      <StickZone label="Look" onChange={(s) => { look.current = s; push(); }} />
    </div>
  );
}

function StickZone({ label, onChange }: { label: string; onChange: (s: Stick) => void }) {
  const base = useRef<HTMLDivElement>(null);
  const knob = useRef<HTMLDivElement>(null);
  const active = useRef<number | null>(null);

  const update = (e: PointerEvent<HTMLDivElement>) => {
    const r = (base.current as HTMLDivElement).getBoundingClientRect();
    const dx = e.clientX - (r.left + r.width / 2);
    const dy = e.clientY - (r.top + r.height / 2);
    const len = Math.hypot(dx, dy);
    const k = len > RADIUS ? RADIUS / len : 1;
    if (knob.current) knob.current.style.transform = `translate(${dx * k}px, ${dy * k}px)`;
    onChange(stickVector(dx, dy, RADIUS));
  };
  const end = (e: PointerEvent<HTMLDivElement>) => {
    if (active.current !== e.pointerId) return;
    active.current = null;
    if (knob.current) knob.current.style.transform = "translate(0px, 0px)";
    onChange(NO_STICK);
  };

  return (
    <div
      ref={base}
      aria-hidden="true"
      className="pointer-events-auto relative grid size-[7.5rem] touch-none select-none place-items-center rounded-full border border-slate-400/25 bg-slate-900/30"
      onPointerDown={(e) => {
        if (active.current !== null) return;
        active.current = e.pointerId;
        e.currentTarget.setPointerCapture(e.pointerId);
        update(e);
      }}
      onPointerMove={(e) => {
        if (active.current === e.pointerId) update(e);
      }}
      onPointerUp={end}
      onPointerCancel={end}
    >
      <span className="pointer-events-none absolute top-2 font-mono text-[9px] uppercase tracking-[0.18em] text-slate-500">{label}</span>
      <div ref={knob} className="pointer-events-none size-11 rounded-full border border-cyan-200/50 bg-cyan-200/15" />
    </div>
  );
}
