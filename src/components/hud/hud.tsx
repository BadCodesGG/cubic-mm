"use client";

import { Fragment, useEffect, useId, useMemo, useRef, useState } from "react";
import type { App } from "@/engine/app";
import { describeNeuron, formatCount } from "@/engine/info";
import { FOCUS, NeuronPanel, SummaryPanel, type PanelView } from "./panels";
import { TouchPad } from "./touch-pad";
// --- r3/links ---
import { Search } from "./search";
import { CopyLink } from "./share";
// --- end r3/links ---
import { useCoarsePointer, useHudFrame, useMedia, useReducedMotion, useSelection, useStatus, type FrameSnap } from "./use-hud";

const SHADOW = { textShadow: "0 0 6px #04060b, 0 0 2px #04060b" } as const;

/** Below this width the bottom-left panel starts collapsed and the status line breaks into two lines. */
const NARROW = "(max-width: 899px)";

/** One line of the status: its parts joined by " · ", never split across a wrap. */
function StatusGroup({ parts }: { parts: string[] }) {
  if (parts.length === 0) return null;
  return (
    <span className="block whitespace-nowrap min-[900px]:inline">
      {parts.join(" · ")}
    </span>
  );
}

function StatusLine({ app }: { app: App }) {
  const s = useStatus(app);
  // Two lines on a narrow screen (the data, then the renderer), one line from 900px up. On WebGL2 the
  // badge below names the renderer, so it is left out of the line.
  const data = [`${formatCount(s.neurons)} neurons`, s.synapses !== null ? `${formatCount(s.synapses)} synapses` : null].filter(
    (p): p is string => p !== null,
  );
  const engine = [s.gpu ? ["WebGPU", s.tier].filter(Boolean).join(" ") : null, s.frameMs > 0 ? `${s.frameMs.toFixed(1)} ms/frame` : null].filter(
    (p): p is string => p !== null,
  );
  return (
    <>
      <p className="mt-1 font-mono text-[11px] tracking-wide text-slate-400" style={SHADOW}>
        <StatusGroup parts={data} />
        {engine.length > 0 ? (
          <Fragment>
            <span className="hidden min-[900px]:inline"> · </span>
            <StatusGroup parts={engine} />
          </Fragment>
        ) : null}
      </p>
      {s.gpu ? null : (
        <p
          title="WebGPU is not in use, so this is the simplified WebGL 2 renderer"
          className="mt-1.5 inline-block rounded-[3px] border border-amber-300/40 bg-amber-300/10 px-1.5 py-0.5 font-mono text-[11px] uppercase tracking-[0.12em] text-amber-100"
        >
          WebGL2 lite mode
        </p>
      )}
    </>
  );
}

function About({ app, open, setOpen }: { app: App; open: boolean; setOpen: (open: boolean) => void }) {
  const button = useRef<HTMLButtonElement>(null);
  const panelId = useId();
  const credits = app.data.manifest.credits;

  useEffect(() => {
    if (!open) return;
    // Capture phase, so Esc closes this panel before the picker sees it and clears the selection.
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== "Escape") return;
      e.preventDefault();
      setOpen(false);
      button.current?.focus();
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [open, setOpen]);

  return (
    <>
      <div data-hud-panel="about" className="pointer-events-auto absolute right-4 top-4">
        <button
          ref={button}
          type="button"
          aria-expanded={open}
          aria-controls={panelId}
          onClick={() => setOpen(!open)}
          className={`h-9 rounded-[4px] border border-slate-500/30 bg-slate-900/40 px-3 text-[11px] tracking-wide text-slate-200 backdrop-blur-sm transition-colors hover:border-slate-400/50 hover:bg-slate-800/50 ${FOCUS}`}
        >
          About the data
        </button>
      </div>
      {open ? (
        // Below the sound button, and capped so the status line, the sticks and the cell panel stay clear: a long
        // citation scrolls inside the panel.
        <div
          id={panelId}
          data-hud-panel="about-panel"
          role="region"
          aria-label="About the data"
          tabIndex={0}
          className={`pointer-events-auto absolute right-4 top-28 max-h-[calc(100dvh-17.5rem)] w-[22rem] max-w-[calc(100vw-2rem)] overflow-y-auto overscroll-contain rounded-[4px] border border-slate-500/25 bg-[#04060b]/85 p-3.5 text-[12px] leading-relaxed text-slate-300 backdrop-blur-sm min-[900px]:max-h-[calc(100dvh-8.5rem)] ${FOCUS}`}
        >
          <p className="text-slate-100">{credits.dataset}</p>
          <p className="mt-1 text-slate-400">
            Licence {credits.licence}.{" "}
            <a href={credits.url} target="_blank" rel="noreferrer" className={`text-cyan-200 underline decoration-cyan-200/40 underline-offset-2 ${FOCUS}`}>
              microns-explorer.org
            </a>
          </p>
          <p className="mt-2.5 font-mono text-[11px] uppercase tracking-[0.12em] text-slate-400">
            {credits.citations.length > 1 ? "Citations" : "Citation"}
          </p>
          {credits.citations.map((c) => (
            <p key={c} className="mt-1 select-text text-slate-300">
              {c}
            </p>
          ))}
          <p className="mt-2.5 text-slate-400">
            Every number in this view is measured from the {formatCount(app.data.neurons.count)} loaded skeletons
            ({app.data.lod === "hi" ? "full" : "simplified"} detail).
          </p>
        </div>
      ) : null}
    </>
  );
}

function ScaleAndCompass({ frame }: { frame: FrameSnap }) {
  return (
    <div className="flex items-end gap-5" style={SHADOW} aria-hidden="true">
      <div className="flex items-center gap-2 pb-0.5">
        <svg width="26" height="26" viewBox="-13 -13 26 26" className="text-slate-300" style={{ opacity: 0.35 + 0.65 * frame.piaLength }}>
          <g transform={`rotate(${frame.piaDeg})`}>
            <path d="M0 10 V-9 M-4 -5 L0 -10 L4 -5" fill="none" stroke="currentColor" strokeWidth="1.25" strokeLinecap="round" strokeLinejoin="round" />
          </g>
        </svg>
        <span className="font-mono text-[11px] uppercase tracking-[0.12em] text-slate-400">pia</span>
      </div>
      <div className="flex flex-col items-end gap-1">
        <span className="font-mono text-[12px] text-slate-300">
          {frame.scaleUm >= 1000 ? `${frame.scaleUm / 1000} mm` : `${frame.scaleUm} µm`}
        </span>
        <div className="h-1.5 border-x border-b border-slate-200/80" style={{ width: Math.max(0, frame.scalePx) }} />
      </div>
    </div>
  );
}

function Hint({ coarse }: { coarse: boolean }) {
  const reduced = useReducedMotion();
  const [phase, setPhase] = useState<"shown" | "fading" | "gone">("shown");

  useEffect(() => {
    const fade = () => setPhase((p) => (p === "shown" ? (reduced ? "gone" : "fading") : p));
    const timer = window.setTimeout(fade, 8000);
    window.addEventListener("keydown", fade, { once: true });
    window.addEventListener("pointerdown", fade, { once: true });
    return () => {
      window.clearTimeout(timer);
      window.removeEventListener("keydown", fade);
      window.removeEventListener("pointerdown", fade);
    };
  }, [reduced]);

  useEffect(() => {
    if (phase !== "fading") return;
    const id = window.setTimeout(() => setPhase("gone"), 1000);
    return () => window.clearTimeout(id);
  }, [phase]);

  if (phase === "gone") return null;
  return (
    <p
      data-hud-panel="hint"
      className={`pointer-events-none absolute inset-x-0 top-16 mx-auto w-fit max-w-[92vw] rounded-full bg-[#04060b]/60 px-4 py-1.5 text-center text-[12px] tracking-wide text-slate-200 backdrop-blur-sm transition-opacity duration-1000 md:top-5 pointer-coarse:top-28 ${
        phase === "fading" ? "opacity-0" : "opacity-100"
      }`}
    >
      {coarse
        ? "Tap a cell. Left thumb moves, right thumb looks."
        : "Click a cell. Space stimulates it. R rides its next spike. WASD flies, Esc frees the mouse."}
    </p>
  );
}

/** The ring around the selected soma, labelled with its type and layer, and a crosshair while the mouse is captured. */
function Markers({ app, frame, selected }: { app: App; frame: FrameSnap; selected: number }) {
  const info = useMemo(() => (selected >= 0 ? describeNeuron(app.data, selected) : null), [app, selected]);
  const showRing = info && frame.ring && frame.ride !== "riding" && frame.ride !== "easing";
  return (
    <>
      {showRing && frame.ring ? (
        <div aria-hidden="true" className="absolute left-0 top-0" style={{ transform: `translate(${frame.ring[0]}px, ${frame.ring[1]}px)` }}>
          <div className="-ml-[19px] -mt-[19px] size-[38px] rounded-full border-[1.5px] border-cyan-100/90 shadow-[0_0_16px_rgba(165,243,252,0.55),inset_0_0_10px_rgba(165,243,252,0.25)]" />
          <span
            className="absolute left-[26px] top-[-8px] whitespace-nowrap font-mono text-[11px] uppercase tracking-[0.12em] text-cyan-50"
            style={SHADOW}
          >
            {info.typeCode || "cell"} · L{info.layer || "?"}
          </span>
        </div>
      ) : null}
      {frame.locked ? (
        <div aria-hidden="true" className="absolute left-1/2 top-1/2 size-1 -translate-x-1/2 -translate-y-1/2 rounded-full bg-slate-100/70" />
      ) : null}
    </>
  );
}

export function Hud({ app }: { app: App }) {
  const selected = useSelection(app);
  const frame = useHudFrame(app);
  const coarse = useCoarsePointer();
  const narrow = useMedia(NARROW);
  const [aboutOpen, setAboutOpen] = useState(false);
  // The selection the visitor expanded the panel for: picking another cell (or none) collapses it again.
  const [expandedFor, setExpandedFor] = useState<number | null>(null);
  const expanded = expandedFor === selected;
  const view: PanelView = {
    compact: narrow && !expanded,
    onToggle: narrow ? () => setExpandedFor(expanded ? null : selected) : null,
  };

  return (
    <div className="pointer-events-none absolute inset-0 z-10">
      <div aria-hidden="true" className="absolute inset-x-0 top-0 h-36 bg-gradient-to-b from-[#04060b]/70 to-transparent" />
      <div aria-hidden="true" className="absolute inset-x-0 bottom-0 h-56 bg-gradient-to-t from-[#04060b]/75 to-transparent" />
      <Markers app={app} frame={frame} selected={selected} />

      {/* On a narrow screen the status wraps short of the About and sound buttons in the corner. */}
      <header data-hud-panel="status" className="absolute left-5 right-40 top-5 sm:right-auto">
        <h1 className="font-mono text-[11px] uppercase tracking-[0.28em] text-slate-200" style={SHADOW}>
          One Cubic Millimetre
        </h1>
        <StatusLine app={app} />
      </header>

      <About app={app} open={aboutOpen} setOpen={setAboutOpen} />

      {/* On a narrow screen the open About panel takes the room the cell panel would use. */}
      <div
        data-hud-panel="cell"
        className={`pointer-events-auto absolute bottom-5 left-5 rounded-[4px] border-l border-cyan-200/25 bg-[#04060b]/60 py-2 pl-3.5 pr-3 backdrop-blur-sm pointer-coarse:bottom-44 ${
          aboutOpen && narrow ? "hidden" : ""
        }`}
      >
        {selected >= 0 ? (
          <NeuronPanel app={app} neuron={selected} ride={frame.ride} rideNeuron={frame.rideNeuron} view={view} actions={<CopyLink app={app} />} /* r3/links */ />
        ) : (
          <SummaryPanel app={app} view={view} actions={<CopyLink app={app} />} /* r3/links */ />
        )}
      </div>

      {aboutOpen ? null : (
        <div data-hud-panel="scale" className="absolute bottom-6 right-6 pointer-coarse:bottom-auto pointer-coarse:right-4 pointer-coarse:top-44">
          <ScaleAndCompass frame={frame} />
        </div>
      )}

      <Hint coarse={coarse} />
      {/* r3/links */}
      <Search app={app} />
      {coarse ? <TouchPad app={app} /> : null}
    </div>
  );
}
