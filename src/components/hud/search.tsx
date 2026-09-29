"use client";

import { useEffect, useId, useMemo, useRef, useState, type KeyboardEvent } from "react";
import type { App } from "@/engine/app";
import { formatLength } from "@/engine/info";
import { QUICK_QUERIES, nearestMatch, search } from "@/engine/search";
import { FOCUS } from "./panels";

/** Where a key press is typing, not commanding. */
const EDITABLE = "input, textarea, select, [contenteditable]";

/**
 * Cell search at the top centre. `/` focuses it, Esc leaves it, Enter (or a click) flies to a result,
 * and Tab or the arrows step through the list. The chips under the empty box each fly to the nearest
 * cell of a kind. The engine does the flying: this only emits `jump`.
 */
export function Search({ app }: { app: App }) {
  const [query, setQuery] = useState("");
  const [focused, setFocused] = useState(false);
  const [active, setActive] = useState(0);
  // Bumped on focus, so the distances are measured from wherever the camera is now.
  const [stamp, setStamp] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const listId = useId();

  // The camera position is sampled when the box gains focus (`stamp`), not on every keystroke.
  const origin = useMemo(() => [...app.cameraPosition()] as [number, number, number], [app, stamp]);
  const result = useMemo(() => search(app.data, query, origin), [app, query, origin]);
  // A chip for a kind of cell this dataset does not hold would do nothing, so it is disabled.
  const chips = useMemo(() => QUICK_QUERIES.map((q) => ({ q, found: search(app.data, q, [0, 0, 0], { limit: 1 }).total > 0 })), [app]);
  const { hits, total } = result;
  const current = Math.min(active, Math.max(0, hits.length - 1));
  const open = focused && query.trim() !== "";

  useEffect(() => {
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key !== "/" || e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.target instanceof Element && e.target.closest(EDITABLE)) return;
      e.preventDefault();
      // A locked mouse cannot reach the results.
      if (document.pointerLockElement) document.exitPointerLock();
      input.current?.focus();
      input.current?.select();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const jump = (neuron: number) => {
    app.bus.emit("jump", { neuron });
    input.current?.blur();
  };

  const runChip = (q: string) => {
    const hit = nearestMatch(app.data, q, app.cameraPosition(), app.selection());
    setQuery("");
    if (hit) jump(hit.neuron);
  };

  const step = (by: number) => setActive((current + by + hits.length) % hits.length);

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    // Typing is not a command: keep R, WASD and Esc away from the ride, the camera and the picker,
    // which listen on `window`.
    e.stopPropagation();
    if (e.key === "Escape") {
      input.current?.blur();
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (hits[current]) jump(hits[current].neuron);
    } else if (hits.length > 0 && (e.key === "ArrowDown" || (e.key === "Tab" && !e.shiftKey))) {
      e.preventDefault();
      step(1);
    } else if (hits.length > 0 && (e.key === "ArrowUp" || (e.key === "Tab" && e.shiftKey))) {
      e.preventDefault();
      step(-1);
    }
  };

  // A press on a chip or a row must not take focus from the box, or the list would close under the click.
  const keepFocus = (e: { preventDefault(): void }) => e.preventDefault();

  return (
    <div
      data-hud-panel="search"
      className="pointer-events-auto absolute left-1/2 top-[7rem] w-[min(21rem,calc(100vw-2rem))] -translate-x-1/2 md:top-16 pointer-coarse:left-5 pointer-coarse:top-[14.5rem] pointer-coarse:w-[calc(100vw-2.5rem)] pointer-coarse:translate-x-0"
    >
      <div className="relative">
        <input
          ref={input}
          type="text"
          role="combobox"
          aria-label="Search cells"
          aria-expanded={open}
          aria-controls={listId}
          aria-activedescendant={open && hits.length > 0 ? `${listId}-${current}` : undefined}
          aria-autocomplete="list"
          autoComplete="off"
          spellCheck={false}
          placeholder="Find a cell"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setActive(0);
          }}
          onFocus={() => {
            setFocused(true);
            setStamp((s) => s + 1);
          }}
          onBlur={() => setFocused(false)}
          onKeyDown={onKeyDown}
          className={`h-9 w-full rounded-[4px] border border-slate-500/30 bg-[#04060b]/60 pl-3 pr-8 text-[12px] text-slate-100 backdrop-blur-sm placeholder:text-slate-400 focus:border-cyan-200/50 pointer-coarse:h-11 pointer-coarse:text-[16px] ${FOCUS}`}
        />
        <kbd className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 rounded-[3px] border border-slate-500/40 px-1 font-mono text-[11px] leading-4 text-slate-400 pointer-coarse:hidden">
          /
        </kbd>
      </div>

      {focused && query === "" ? (
        <div className="mt-1.5 flex flex-wrap gap-1.5">
          {chips.map(({ q, found }) => (
            <button
              key={q}
              type="button"
              disabled={!found}
              onMouseDown={keepFocus}
              onClick={() => runChip(q)}
              className={`rounded-full border border-slate-500/30 bg-[#04060b]/60 px-2.5 py-1 text-[11px] tracking-wide text-slate-300 backdrop-blur-sm transition-colors hover:border-slate-400/50 hover:bg-slate-800/50 disabled:cursor-not-allowed disabled:opacity-40 pointer-coarse:py-2 ${FOCUS}`}
            >
              {q}
            </button>
          ))}
        </div>
      ) : null}

      {open ? (
        <div className="mt-1.5 overflow-hidden rounded-[4px] border border-slate-500/25 bg-[#04060b]/85 text-[12px] backdrop-blur-sm">
          {hits.length > 0 ? (
            <ul id={listId} role="listbox" aria-label="Matching cells">
              {hits.map((h, i) => (
                <li
                  key={h.neuron}
                  id={`${listId}-${i}`}
                  role="option"
                  aria-selected={i === current}
                  onMouseDown={keepFocus}
                  onMouseEnter={() => setActive(i)}
                  onClick={() => jump(h.neuron)}
                  className={`cursor-pointer px-3 py-1.5 ${i === current ? "bg-cyan-200/10" : ""}`}
                >
                  <div className="flex items-baseline justify-between gap-3">
                    <span className="min-w-0 truncate text-slate-100">
                      <span className="mr-2 font-mono text-[11px] uppercase tracking-[0.12em] text-slate-400">{h.typeCode || "?"}</span>
                      <span className="first-letter:uppercase">{h.typeName}</span>
                    </span>
                    <span className="shrink-0 font-mono text-[11px] tabular-nums text-slate-300">{formatLength(h.distanceUm)}</span>
                  </div>
                  <div className="flex items-baseline gap-2 font-mono text-[11px] text-slate-400">
                    <span>L{h.layer || "?"}</span>
                    <span className="truncate">{h.rootId}</span>
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <p id={listId} className="px-3 py-2 text-slate-400">
              No cell matches. Try a type code (4P, BC), a layer, or a root id.
            </p>
          )}
          {hits.length > 0 ? (
            <div role="status" className="flex items-center justify-between border-t border-slate-500/20 px-3 py-1 text-[11px] text-slate-400">
              <span>{total > hits.length ? `Nearest ${hits.length} of ${total}` : `${total} ${total === 1 ? "match" : "matches"}`}</span>
              {hits.length > 1 ? (
                <button
                  type="button"
                  onMouseDown={keepFocus}
                  onClick={() => step(1)}
                  className={`rounded-[3px] px-1.5 py-0.5 text-slate-300 hover:bg-slate-800/50 ${FOCUS}`}
                >
                  Next <kbd className="ml-1 font-mono text-slate-400 pointer-coarse:hidden">Tab</kbd>
                </button>
              ) : null}
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
