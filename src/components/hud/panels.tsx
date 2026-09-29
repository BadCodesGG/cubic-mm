"use client";

import { useMemo, type ReactNode } from "react";
import type { App } from "@/engine/app";
import { datasetSummary, describeNeuron, formatCount, formatLength, piaDepth } from "@/engine/info";
import type { RideState } from "@/engine/camera/ride";

export const FOCUS = "outline-none focus-visible:ring-2 focus-visible:ring-cyan-300/80 focus-visible:ring-offset-1 focus-visible:ring-offset-[#04060b]";

/**
 * How much of a bottom-left panel to show. On a narrow screen the panel starts `compact` (headline and one
 * line) and `onToggle` flips it; on a wide screen `onToggle` is null and the panel is always in full.
 */
export interface PanelView {
  compact: boolean;
  onToggle: (() => void) | null;
}

function Row({ label, stacked, children }: { label: string; stacked?: boolean; children: ReactNode }) {
  return (
    <div className={stacked ? "flex flex-col gap-1" : "flex items-baseline gap-3"}>
      <dt className="w-[5.6rem] shrink-0 font-mono text-[11px] uppercase tracking-[0.12em] text-slate-400">{label}</dt>
      <dd className="min-w-0 text-slate-200">{children}</dd>
    </div>
  );
}

function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="ml-1.5 rounded-[3px] border border-slate-500/40 px-1 font-mono text-[11px] leading-4 text-slate-400 pointer-coarse:hidden">
      {children}
    </kbd>
  );
}

export function ActionButton({
  onClick,
  children,
  primary,
  pressed,
}: {
  onClick: () => void;
  children: ReactNode;
  primary?: boolean;
  pressed?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={pressed}
      className={`inline-flex items-center rounded-[4px] border px-2.5 py-1.5 text-[11px] tracking-wide transition-colors pointer-coarse:px-3 pointer-coarse:py-2.5 ${FOCUS} ${
        primary
          ? "border-amber-300/50 bg-amber-300/10 text-amber-100 hover:bg-amber-300/20"
          : "border-slate-500/30 bg-slate-900/40 text-slate-300 hover:border-slate-400/50 hover:bg-slate-800/50"
      }`}
    >
      {children}
    </button>
  );
}

/** Show or hide a compact panel's detail rows; renders nothing on a wide screen. */
function DetailToggle({ view }: { view: PanelView }) {
  if (!view.onToggle) return null;
  return (
    <button
      type="button"
      aria-expanded={!view.compact}
      onClick={view.onToggle}
      className={`ml-auto rounded-[4px] border border-slate-500/30 bg-slate-900/40 px-2 py-1 text-[11px] tracking-wide text-slate-300 transition-colors hover:border-slate-400/50 hover:bg-slate-800/50 pointer-coarse:py-2 ${FOCUS}`}
    >
      {view.compact ? "Show details" : "Hide details"}
    </button>
  );
}

function rideNote(app: App, ride: RideState, neuron: number, rideNeuron: number): string | null {
  if (ride === "waiting") return "Waiting for its next spike. Space fires it now.";
  if (ride === "easing") return "Easing out.";
  if (ride !== "riding") return null;
  const into = rideNeuron !== neuron && rideNeuron >= 0 ? ` into a ${describeNeuron(app.data, rideNeuron).typeName}` : "";
  return `Riding the pulse${into}. Esc releases.`;
}

export function NeuronPanel({
  app,
  neuron,
  ride,
  rideNeuron,
  view,
  actions,
}: {
  app: App;
  neuron: number;
  ride: RideState;
  rideNeuron: number;
  view: PanelView;
  /** Extra buttons at the end of the action row. */
  actions?: ReactNode;
}) {
  const info = useMemo(() => describeNeuron(app.data, neuron), [app, neuron]);
  const pia = useMemo(() => piaDepth(app.data), [app]);
  const excit = info.className === "excitatory";
  const busy = ride !== "idle";
  const note = rideNote(app, ride, neuron, rideNeuron);
  const emit = (type: "stimulate" | "ride") => app.bus.emit(type, { neuron });

  return (
    <section aria-label="Selected neuron" className="w-[19.5rem] max-w-[calc(100vw-2rem)]">
      <div className="flex items-center gap-2">
        <span
          className={`rounded-[3px] px-1.5 py-0.5 font-mono text-[11px] uppercase tracking-[0.12em] ${
            excit ? "bg-amber-300/15 text-amber-200" : "bg-sky-300/15 text-sky-200"
          }`}
        >
          {info.className}
        </span>
        <span className="font-mono text-[11px] uppercase tracking-[0.12em] text-slate-400">{info.typeCode || "no code"}</span>
        <DetailToggle view={view} />
      </div>
      <h2 className="mt-1.5 text-[15px] font-medium leading-tight text-slate-100 first-letter:uppercase">{info.typeName}</h2>

      {view.compact ? (
        <p className="mt-1 text-[12px] leading-snug text-slate-300">
          {info.layerName} · {formatLength(info.depthBelowPiaUm)} below the pia
        </p>
      ) : (
        <dl className="mt-2.5 space-y-1 text-[12px] leading-snug">
          <Row label="Layer">{info.layerName.replace("Layer ", "")}</Row>
          <Row label="Depth">
            {formatLength(info.depthBelowPiaUm)} <span className="text-slate-400">below the pia</span>
          </Row>
          <Row label="Cable">
            {formatLength(info.cableLengthUm)}
            <span className="block text-[12px] text-slate-400">
              axon {formatLength(info.axonLengthUm)} · dendrite {formatLength(info.dendriteLengthUm)}
            </span>
          </Row>
          <Row label="Synapses">
            {info.outgoing === null || info.incoming === null ? (
              <span className="text-slate-400">table not loaded yet</span>
            ) : (
              <>
                {formatCount(info.outgoing)} <span className="text-slate-400">out</span> · {formatCount(info.incoming)}{" "}
                <span className="text-slate-400">in</span>
              </>
            )}
          </Row>
          <Row label="Root id">
            <span className="select-text font-mono text-[11px] text-slate-300">{info.rootId}</span>
          </Row>
        </dl>
      )}
      <p className="sr-only">Pia depth is estimated from {pia.source === "layer-fit" ? "a layer fit to soma depths" : "the shallowest soma"}.</p>

      <div className="mt-3 flex flex-wrap gap-1.5">
        <ActionButton primary onClick={() => emit("stimulate")}>
          Stimulate <Kbd>Space</Kbd>
        </ActionButton>
        <ActionButton
          pressed={busy}
          onClick={() => {
            if (busy) app.cancelRide();
            else emit("ride");
          }}
        >
          {busy ? "Cancel ride" : "Ride the spike"} {busy ? null : <Kbd>R</Kbd>}
        </ActionButton>
        <ActionButton onClick={() => app.bus.emit("select", { neuron: -1 })}>
          Deselect <Kbd>Esc</Kbd>
        </ActionButton>
        {actions}
      </div>
      <p className="mt-2 min-h-4 text-[11px] text-cyan-200/80" role="status">
        {note}
      </p>
    </section>
  );
}

export function SummaryPanel({ app, view, actions }: { app: App; view: PanelView; actions?: ReactNode }) {
  const s = useMemo(() => datasetSummary(app.data), [app]);
  const approx = s.lod === "lite" ? "about " : "";
  return (
    <section aria-label="Dataset summary" className="w-[19.5rem] max-w-[calc(100vw-2rem)]">
      <div className="flex items-center gap-2">
        <p className="font-mono text-[11px] uppercase tracking-[0.12em] text-slate-400">The volume</p>
        <DetailToggle view={view} />
      </div>
      <h2 className="mt-1 text-[15px] font-medium leading-tight text-slate-100">
        {formatCount(s.neuronCount)} real neurons
        {s.synapseCount !== null ? `, ${formatCount(s.synapseCount)} synapses` : ""}
      </h2>
      {view.compact ? null : (
        <dl className="mt-2.5 space-y-1 text-[12px] leading-snug">
          <Row label="Classes">
            {formatCount(s.excitatory)} <span className="text-slate-400">excitatory</span> · {formatCount(s.inhibitory)}{" "}
            <span className="text-slate-400">inhibitory</span>
          </Row>
          <Row label="Cable">
            {approx}
            {formatLength(s.totalCableKm * 1e9)}
            <span className="block text-[12px] text-slate-400">of axon and dendrite, end to end</span>
          </Row>
          <Row label="Types" stacked>
            <ul className="grid grid-cols-3 gap-x-5 gap-y-0.5 font-mono text-[12px]">
              {s.perType.map((t) => (
                <li key={t.code || "none"} title={t.name} className="flex justify-between gap-2 whitespace-nowrap">
                  <span className="text-slate-300">{t.code || "?"}</span>
                  <span className="tabular-nums text-slate-400">{t.count}</span>
                </li>
              ))}
            </ul>
          </Row>
        </dl>
      )}
      <p className="mt-2 text-[12px] text-slate-400">Click a cell to read its numbers.</p>
      {actions ? <div className="mt-2.5 flex flex-wrap gap-1.5">{actions}</div> : null}
    </section>
  );
}
