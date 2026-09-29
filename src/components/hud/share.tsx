"use client";

import { useEffect, useRef, useState } from "react";
import type { App } from "@/engine/app";
import { ActionButton } from "./panels";

const COPIED_MS = 1500;

/**
 * Copies a link to the current view. The URL hash already follows the camera and selection
 * (engine/share.ts); `app.shareUrl()` writes it once more so the copy is never up to two seconds stale.
 * Where the clipboard is refused (an insecure origin, a denied permission) the link is shown in a
 * read-only, already-selected field to copy by hand.
 */
export function CopyLink({ app }: { app: App }) {
  const [copied, setCopied] = useState(false);
  const [manual, setManual] = useState<string | null>(null);
  const timer = useRef(0);
  const field = useRef<HTMLInputElement>(null);

  useEffect(() => () => window.clearTimeout(timer.current), []);
  useEffect(() => {
    if (manual === null) return;
    field.current?.focus();
    field.current?.select();
  }, [manual]);

  const copy = () => {
    const url = app.shareUrl();
    const done = () => {
      setManual(null);
      setCopied(true);
      window.clearTimeout(timer.current);
      timer.current = window.setTimeout(() => setCopied(false), COPIED_MS);
    };
    const refused = () => setManual(url);
    // The write starts inside the click handler, which is what browsers require of it.
    if (!navigator.clipboard?.writeText) refused();
    else navigator.clipboard.writeText(url).then(done, refused);
  };

  return (
    <>
      <ActionButton onClick={copy}>{copied ? "Copied" : "Copy link"}</ActionButton>
      <span role="status" className="sr-only">
        {copied ? "Link copied to the clipboard" : ""}
      </span>
      {manual !== null ? (
        <input
          ref={field}
          readOnly
          value={manual}
          aria-label="Link to this view, press Ctrl+C to copy"
          onFocus={(e) => e.currentTarget.select()}
          onBlur={() => setManual(null)}
          // Typing here is not a command: keep R, WASD and Esc away from the ride, the camera and the picker.
          onKeyDown={(e) => e.stopPropagation()}
          className="block w-full basis-full select-text rounded-[4px] border border-slate-500/40 bg-[#04060b]/70 px-2 py-1 font-mono text-[11px] text-slate-200 outline-none focus-visible:ring-2 focus-visible:ring-cyan-300/80"
        />
      ) : null}
    </>
  );
}
