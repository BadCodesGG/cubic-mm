"use client";

import { useEffect, useMemo, useState } from "react";
import type { App } from "@/engine/app";
import { describeNeuron } from "@/engine/info";
import { captionAt, tourCaptions } from "@/engine/tour";
import { FOCUS } from "./panels";
import { useCoarsePointer } from "./use-hud";

const SHADOW = { textShadow: "0 0 10px #04060b, 0 0 3px #04060b" } as const;

/** The caption on screen now, polled from the tour's clock each animation frame (it changes six times a tour). */
function useCaption(app: App): number {
  const [index, setIndex] = useState(() => captionAt(app.tour.time()));
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      setIndex(captionAt(app.tour.time()));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [app]);
  return index;
}

/** What the HUD shows while the tour plays: one caption line at a time, and Skip. */
export function TourOverlay({ app }: { app: App }) {
  const coarse = useCoarsePointer();
  const captions = useMemo(
    () => tourCaptions(describeNeuron(app.data, app.tour.heroNeuron).depthBelowPiaUm, coarse),
    [app, coarse],
  );
  const index = useCaption(app);
  // Keep the last line in place while it fades out, rather than blanking it at once.
  const [shown, setShown] = useState(Math.max(0, index));
  if (index >= 0 && index !== shown) setShown(index);

  return (
    <div data-hud-panel="tour" className="pointer-events-none absolute inset-0">
      <div aria-hidden="true" className="absolute inset-x-0 bottom-0 h-56 bg-gradient-to-t from-[#04060b]/70 to-transparent" />
      <p
        aria-live="polite"
        className={`absolute inset-x-0 bottom-24 mx-auto max-w-[40rem] px-6 text-center text-[17px] leading-snug text-slate-50 transition-opacity duration-700 ease-out sm:text-[19px] ${
          index >= 0 ? "opacity-100" : "opacity-0"
        }`}
        style={SHADOW}
      >
        {captions[shown]}
      </p>
      <button
        type="button"
        onClick={() => app.tour.skip()}
        className={`pointer-events-auto absolute bottom-8 left-1/2 h-9 -translate-x-1/2 rounded-[4px] border border-slate-500/30 bg-slate-900/40 px-4 text-[11px] uppercase tracking-[0.16em] text-slate-300 backdrop-blur-sm transition-colors hover:border-slate-400/50 hover:bg-slate-800/50 hover:text-slate-100 ${FOCUS}`}
      >
        Skip
      </button>
    </div>
  );
}

/** "Replay the intro", for the About panel. */
export function ReplayIntro({ app, onReplay }: { app: App; onReplay: () => void }) {
  return (
    <button
      type="button"
      onClick={() => {
        onReplay();
        app.tour.replay();
      }}
      className={`mt-3 h-8 rounded-[4px] border border-slate-500/30 bg-slate-900/40 px-3 text-[11px] tracking-wide text-slate-200 transition-colors hover:border-slate-400/50 hover:bg-slate-800/50 ${FOCUS}`}
    >
      Replay the intro
    </button>
  );
}
