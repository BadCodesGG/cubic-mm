"use client";

import { useEffect, useState } from "react";
import type { App } from "@/engine/app";

const TOAST_MS = 1500;

/** Camera button beside the sound toggle: saves the current view as a PNG at twice the resolution (`P`). */
export function ScreenshotButton({ app }: { app: App }) {
  return (
    <button
      type="button"
      title="Save a screenshot (P)"
      aria-label="Save a screenshot"
      onClick={() => app.bus.emit("screenshot", {})}
      className="pointer-events-auto absolute right-[3.75rem] top-14 flex size-9 items-center justify-center rounded-[4px] border border-slate-500/30 bg-slate-900/40 text-slate-200 backdrop-blur-sm transition-colors hover:bg-slate-800/60 hover:text-cyan-100 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-cyan-200/80"
    >
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M4 8h3l1.5-2h7L17 8h3v11H4z" />
        <circle cx="12" cy="13" r="3.5" />
      </svg>
    </button>
  );
}

/** "Saved" for 1.5 s after each screenshot (or a short failure note). */
export function ScreenshotToast({ app }: { app: App }) {
  const [message, setMessage] = useState<string | null>(null);
  useEffect(() => {
    let timer = 0;
    const off = app.onScreenshot((filename) => {
      setMessage(filename ? "Saved" : "Could not save the screenshot");
      window.clearTimeout(timer);
      timer = window.setTimeout(() => setMessage(null), TOAST_MS);
    });
    return () => {
      off();
      window.clearTimeout(timer);
    };
  }, [app]);
  return (
    <p
      role="status"
      className={`pointer-events-none absolute right-4 top-[6.25rem] rounded-[4px] bg-[#04060b]/75 px-3 py-1.5 font-mono text-[11px] tracking-[0.12em] text-cyan-50 backdrop-blur-sm transition-opacity duration-300 ${
        message ? "opacity-100" : "opacity-0"
      }`}
    >
      {message ?? ""}
    </p>
  );
}
