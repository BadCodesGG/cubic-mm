import type { ReactNode } from "react";

export const LINK =
  "text-cyan-200 underline decoration-cyan-200/40 underline-offset-2 outline-none hover:decoration-cyan-200 focus-visible:ring-2 focus-visible:ring-cyan-300/80 focus-visible:ring-offset-1 focus-visible:ring-offset-[#04060b]";

export function Section({ id, kicker, title, children }: { id: string; kicker: string; title: string; children: ReactNode }) {
  return (
    <section aria-labelledby={`${id}-title`} className="border-t border-slate-500/20 py-10">
      <p className="font-mono text-[10px] uppercase tracking-[0.28em] text-slate-500">{kicker}</p>
      <h2 id={`${id}-title`} className="mt-2 text-2xl font-semibold tracking-tight text-slate-100">
        {title}
      </h2>
      <div className="mt-4 space-y-4 text-[15px] leading-relaxed text-slate-300">{children}</div>
    </section>
  );
}

export function Stat({ value, label }: { value: string; label: string }) {
  return (
    <div className="rounded-[4px] border border-slate-500/25 bg-slate-900/30 px-3.5 py-3">
      <dd className="font-mono text-xl text-cyan-100">{value}</dd>
      <dt className="mt-1 font-mono text-[10px] uppercase tracking-[0.14em] text-slate-500">{label}</dt>
    </div>
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="rounded-[3px] border border-slate-500/40 px-1.5 py-0.5 font-mono text-[11px] text-slate-300">{children}</kbd>
  );
}

/** A small table that scrolls sideways at phone width rather than squeezing its columns. */
export function Table({ caption, head, rows }: { caption: string; head: string[]; rows: string[][] }) {
  return (
    <div className="overflow-x-auto rounded-[4px] border border-slate-500/25">
      <table className="w-full min-w-[20rem] text-left text-[13px]">
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr className="border-b border-slate-500/25 font-mono text-[10px] uppercase tracking-[0.14em] text-slate-500">
            {head.map((h) => (
              <th key={h} scope="col" className="px-3 py-2 font-normal">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-500/15">
          {rows.map((r) => (
            <tr key={r[0]}>
              {r.map((cell, i) => (
                <td key={i} className={`px-3 py-2 align-top ${i === 0 ? "font-mono text-slate-200" : i === 1 ? "font-mono text-cyan-100" : "text-slate-400"}`}>
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
