"use client";

import { useEffect, useId, useRef, useState, useSyncExternalStore } from "react";
import type { App } from "@/engine/app";
import { FOCUS } from "./panels";

const SHADOW = { textShadow: "0 0 10px #04060b, 0 0 3px #04060b" } as const;

/** True while a guided story plays. */
export function useStoryRunning(app: App): boolean {
  return useSyncExternalStore(
    (notify) => app.bus.on("story", notify),
    () => app.stories.running(),
    () => false,
  );
}

/** The caption on screen now, polled from the story's clock each animation frame. */
function useStoryCaption(app: App): string | null {
  const [caption, setCaption] = useState(() => app.stories.caption());
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      setCaption(app.stories.caption());
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [app]);
  return caption;
}

/** What the HUD shows while a story plays: its title, one caption at a time, and Skip. */
export function StoryOverlay({ app }: { app: App }) {
  const id = app.stories.id();
  const title = app.stories.list.find((s) => s.id === id)?.title ?? "";
  const caption = useStoryCaption(app);
  // Keep the last line in place while it fades out, rather than blanking it at once.
  const [shown, setShown] = useState(caption ?? "");
  if (caption !== null && caption !== shown) setShown(caption);

  return (
    <div data-hud-panel="story" className="pointer-events-none absolute inset-0">
      <p className="absolute inset-x-0 top-6 text-center font-mono text-[11px] uppercase tracking-[0.24em] text-slate-300" style={SHADOW}>
        {title}
      </p>
      <div aria-hidden="true" className="absolute inset-x-0 bottom-0 h-56 bg-gradient-to-t from-[#04060b]/70 to-transparent" />
      <p
        aria-live="polite"
        className={`absolute inset-x-0 bottom-24 mx-auto max-w-[42rem] px-6 text-center text-[17px] leading-snug text-slate-50 transition-opacity duration-700 ease-out sm:text-[19px] ${
          caption !== null ? "opacity-100" : "opacity-0"
        }`}
        style={SHADOW}
      >
        {shown}
      </p>
      <button
        type="button"
        onClick={() => app.stories.skip()}
        className={`pointer-events-auto absolute bottom-8 left-1/2 h-9 -translate-x-1/2 rounded-[4px] border border-slate-500/30 bg-slate-900/40 px-4 text-[11px] uppercase tracking-[0.16em] text-slate-300 backdrop-blur-sm transition-colors hover:border-slate-400/50 hover:bg-slate-800/50 hover:text-slate-100 ${FOCUS}`}
      >
        Skip
      </button>
    </div>
  );
}

/** The stories as buttons: title, then its one-line blurb. */
function StoryButtons({ app, onPick }: { app: App; onPick: () => void }) {
  return (
    <ul className="space-y-1">
      {app.stories.list.map((s) => (
        <li key={s.id}>
          <button
            type="button"
            onClick={() => {
              onPick();
              app.stories.start(s.id);
            }}
            className={`block w-full rounded-[3px] px-2 py-1.5 text-left transition-colors hover:bg-slate-800/60 ${FOCUS}`}
          >
            <span className="block text-[12px] text-slate-100">{s.title}</span>
            <span className="block text-[11px] leading-snug text-slate-400">{s.blurb}</span>
          </button>
        </li>
      ))}
    </ul>
  );
}

/** "Stories", next to the speed control: opens the list of guided stories. Hidden on phone widths. */
export function StoriesButton({ app }: { app: App }) {
  const [open, setOpen] = useState(false);
  const button = useRef<HTMLButtonElement>(null);
  const box = useRef<HTMLDivElement>(null);
  const panelId = useId();

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== "Escape") return;
      e.preventDefault();
      setOpen(false);
      button.current?.focus();
    };
    const onDown = (e: PointerEvent) => {
      if (e.target instanceof Node && !box.current?.contains(e.target)) setOpen(false);
    };
    window.addEventListener("keydown", onKey, true);
    window.addEventListener("pointerdown", onDown, true);
    return () => {
      window.removeEventListener("keydown", onKey, true);
      window.removeEventListener("pointerdown", onDown, true);
    };
  }, [open]);

  if (app.stories.list.length === 0) return null;
  return (
    // Phones have no room beside the speed control; the About panel lists the stories there.
    <div ref={box} className="pointer-events-auto relative ml-2 mt-2 inline-block align-top max-[519px]:hidden">
      <button
        ref={button}
        type="button"
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen(!open)}
        className={`h-7 rounded-[3px] border px-2 font-mono text-[11px] tracking-wide backdrop-blur-sm transition-colors ${
          open ? "border-cyan-200/50 bg-cyan-200/15 text-cyan-50" : "border-slate-500/30 bg-slate-900/40 text-slate-300 hover:border-slate-400/50 hover:bg-slate-800/50"
        } ${FOCUS}`}
      >
        Stories
      </button>
      {open ? (
        <div
          id={panelId}
          data-hud-panel="stories"
          role="region"
          aria-label="Stories"
          className="absolute left-0 top-full z-20 mt-1.5 w-[20rem] max-w-[calc(100vw-2.5rem)] rounded-[4px] border border-slate-500/25 bg-[#04060b]/85 p-1.5 backdrop-blur-sm"
        >
          <StoryButtons app={app} onPick={() => setOpen(false)} />
        </div>
      ) : null}
    </div>
  );
}

/** The Stories entry in the About panel. */
export function StoryList({ app, onPick }: { app: App; onPick: () => void }) {
  if (app.stories.list.length === 0) return null;
  return (
    <>
      <p className="mt-3 font-mono text-[11px] uppercase tracking-[0.12em] text-slate-400">Stories</p>
      <div className="-mx-2 mt-1">
        <StoryButtons app={app} onPick={onPick} />
      </div>
    </>
  );
}
