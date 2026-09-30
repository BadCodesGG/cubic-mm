"use client";

import { useMemo, useSyncExternalStore } from "react";
import type { App } from "@/engine/app";
import { formatCount } from "@/engine/info";
import { ActionButton, Kbd } from "./panels";

/** Whether the visitor pinned `neuron`'s graph. A timed preview after a search flight is not a pin. */
function usePinned(app: App, neuron: number): boolean {
  return useSyncExternalStore(
    (notify) => {
      const offs = [app.bus.on("partners", notify), app.bus.on("select", notify)];
      return () => offs.forEach((off) => off());
    },
    () => app.wiring.pinned === neuron,
    () => false,
  );
}

const plural = (n: number, one: string, many: string) => `${formatCount(n)} ${n === 1 ? one : many}`;

/**
 * Under the cell panel: the selected cell's partners as a sentence ("118 outputs · 94 inputs (21
 * inhibitory)", coloured like the lines) and the toggle that keeps its graph in the scene. The
 * toggle is the only way to show the graph on a touch screen, which has no hover.
 */
export function WiringPanel({ app, selected, compact }: { app: App; selected: number; compact: boolean }) {
  const pinned = usePinned(app, selected);
  const stats = useMemo(() => (selected >= 0 ? app.partners?.partnerStats(selected) ?? null : null), [app, selected]);
  if (!stats) return null;
  return (
    <section aria-label="Wiring" className="mt-2.5 w-[19.5rem] max-w-[calc(100vw-2rem)] border-t border-slate-500/25 pt-2.5">
      <div className="flex items-center gap-2">
        <span className="font-mono text-[11px] uppercase tracking-[0.12em] text-slate-400">Wiring</span>
        <span className="ml-auto">
          <ActionButton primary={pinned} pressed={pinned} onClick={() => app.bus.emit("partners", { neuron: selected, show: !pinned })}>
            Show wiring <Kbd>G</Kbd>
          </ActionButton>
        </span>
      </div>
      <p className="mt-1.5 text-[12px] leading-snug text-slate-200">
        <span className="text-amber-200">{plural(stats.outputs, "output", "outputs")}</span>
        {" · "}
        <span className="text-cyan-200">{plural(stats.inputs, "input", "inputs")}</span>
        {stats.inhibitoryInputs > 0 ? <span className="text-violet-300"> ({formatCount(stats.inhibitoryInputs)} inhibitory)</span> : null}
      </p>
      {compact ? null : (
        <p className="mt-0.5 text-[12px] leading-snug text-slate-400">
          {plural(stats.outputSynapses, "synapse", "synapses")} out · {formatCount(stats.inputSynapses)} in
        </p>
      )}
    </section>
  );
}
