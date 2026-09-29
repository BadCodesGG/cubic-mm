"use client";

import { useMemo, type ReactNode } from "react";
import type { App } from "@/engine/app";
import { datasetSummary, describeNeuron, formatCount, formatLength, piaDepth } from "@/engine/info";
import type { RideState } from "@/engine/camera/ride";

export const FOCUS = "outline-none focus-visible:ring-2 focus-visible:ring-cyan-300/80 focus-visible:ring-offset-1 focus-visible:ring-offset-[#04060b]";

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-baseline gap-3">
      <dt className="w-[5.6rem] shrink-0 font-mono text-[10px] uppercase tracking-[0.14em] text-slate-500">{label}</dt>
      <dd className="min-w-0 text-slate-200">{children}</dd>
    </div>
  );
}

function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="ml-1.5 rounded-[3px] border border-slate-500/40 px-1 font-mono text-[9px] leading-4 text-slate-400 pointer-coarse:hidden">
      {children}
    </kbd>
  );
}

function ActionButton({
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

function rideNote(app: App, ride: RideState, neuron: number, rideNeuron: number): string | null {
  if (ride === "waiting") return "Waiting for its next spike. Space fires it now.";
  if (ride === "easing") return "Easing out.";
  if (ride !== "riding") return null;
  const into = rideNeuron !== neuron && rideNeuron >= 0 ? ` into a ${describeNeuron(app.data, rideNeuron).typeName}` : "";
  return `Riding the pulse${into}. Esc releases.`;
}

export function NeuronPanel({ app, neuron, ride, rideNeuron }: { app: App; neuron: number; ride: RideState; rideNeuron: number }) {
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
          className={`rounded-[3px] px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-[0.14em] ${
            excit ? "bg-amber-300/15 text-amber-200" : "bg-sky-300/15 text-sky-200"
          }`}
        >
          {info.className}
        </span>
        <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-slate-500">{info.typeCode || "no code"}</span>
      </div>
      <h2 className="mt-1.5 text-[15px] font-medium leading-tight text-slate-100 first-letter:uppercase">{info.typeName}</h2>

      <dl className="mt-2.5 space-y-1 text-[12px] leading-snug">
        <Row label="Layer">{info.layerName.replace("Layer ", "")}</Row>
        <Row label="Depth">
          {formatLength(info.depthBelowPiaUm)} <span className="text-slate-500">below the pia</span>
        </Row>
        <Row label="Cable">
          {formatLength(info.cableLengthUm)}
          <span className="block text-[11px] text-slate-500">
            axon {formatLength(info.axonLengthUm)} · dendrite {formatLength(info.dendriteLengthUm)}
          </span>
        </Row>
        <Row label="Synapses">
          {info.outgoing === null || info.incoming === null ? (
            <span className="text-slate-500">table not loaded yet</span>
          ) : (
            <>
              {formatCount(info.outgoing)} <span className="text-slate-500">out</span> · {formatCount(info.incoming)}{" "}
              <span className="text-slate-500">in</span>
            </>
          )}
        </Row>
        <Row label="Root id">
          <span className="select-text font-mono text-[11px] text-slate-300">{info.rootId}</span>
        </Row>
      </dl>
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
      </div>
      <p className="mt-2 min-h-4 text-[11px] text-cyan-200/80" role="status">
        {note}
      </p>
    </section>
  );
}

export function SummaryPanel({ app }: { app: App }) {
  const s = useMemo(() => datasetSummary(app.data), [app]);
  const approx = s.lod === "lite" ? "about " : "";
  return (
    <section aria-label="Dataset summary" className="w-[19.5rem] max-w-[calc(100vw-2rem)]">
      <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-slate-500">The volume</p>
      <h2 className="mt-1 text-[15px] font-medium leading-tight text-slate-100">
        {formatCount(s.neuronCount)} real neurons
        {s.synapseCount !== null ? `, ${formatCount(s.synapseCount)} synapses` : ""}
      </h2>
      <dl className="mt-2.5 space-y-1 text-[12px] leading-snug">
        <Row label="Classes">
          {formatCount(s.excitatory)} <span className="text-slate-500">excitatory</span> · {formatCount(s.inhibitory)}{" "}
          <span className="text-slate-500">inhibitory</span>
        </Row>
        <Row label="Cable">
          {approx}
          {formatLength(s.totalCableKm * 1e9)}
          <span className="block text-[11px] text-slate-500">of axon and dendrite, end to end</span>
        </Row>
        <Row label="Types">
          <span className="flex flex-wrap gap-x-2.5 font-mono text-[11px] text-slate-300">
            {s.perType.map((t) => (
              <span key={t.code || "none"} title={t.name} className="whitespace-nowrap">
                {t.code || "?"} <span className="text-slate-500">{t.count}</span>
              </span>
            ))}
          </span>
        </Row>
      </dl>
      <p className="mt-2.5 text-[11px] text-slate-500">Click a cell to read its numbers.</p>
    </section>
  );
}
