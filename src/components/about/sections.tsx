import { DEFAULT_PARAMS } from "@/engine/sim/model";
import type { DatasetFacts } from "@/lib/dataset-facts";
import { Kbd, LINK, Section, Stat, Table } from "./section";

const int = new Intl.NumberFormat("en-US");

export function WhatYouAreLookingAt({ facts }: { facts: DatasetFacts }) {
  const [x, y, z] = facts.extentUm;
  return (
    <Section id="what" kicker="01 · The volume" title="What you are looking at">
      <p>
        A cubic millimetre of mouse visual cortex, called minnie65 by the MICrONS project: the only piece of brain ever
        mapped to every synapse. Cell by cell and wire by wire, it is the largest piece anyone has traced.
      </p>
      <dl className="grid grid-cols-2 gap-3">
        <Stat value={int.format(facts.neuronCount)} label="neurons" />
        <Stat value={int.format(facts.synapseCount)} label="synapses between them" />
        <Stat value={`${int.format(Math.round(x))} × ${int.format(Math.round(z))} µm`} label="width × length" />
        <Stat value={`${int.format(Math.round(y))} µm`} label="depth, pia downward" />
      </dl>
      <p>
        {int.format(facts.excitatory)} of the cells are excitatory and {int.format(facts.inhibitory)} inhibitory. Every
        one is a proofread skeleton: a person checked and corrected the automatic tracing.
      </p>
      <Table
        nowrapFirst
        caption="Neurons in this view by cell type"
        head={["Type", "Cells", "What it is"]}
        rows={facts.cellTypes.map((t) => [t.code || "none", int.format(t.count), t.name])}
      />
    </Section>
  );
}

export function TheData({ facts }: { facts: DatasetFacts }) {
  const { credits } = facts;
  return (
    <Section id="data" kicker="02 · Provenance" title="The data">
      <p>
        {credits.dataset}. The skeletons and the cell-type table come straight from the public MICrONS release.
      </p>
      <p>
        The synapse table has about 337 million rows and was streamed once. Of them, 180,707 join two of these cells.
        449 were dropped because the presynaptic cell has no axon in its skeleton, and 23,376 self-synapses (a cell
        onto itself) were dropped because a point neuron would only feed on its own output. That leaves{" "}
        {int.format(facts.synapseCount)} synapses, each with a position, the presynaptic cell&apos;s path distance
        along its axon, and a cleft size.
      </p>
      <p>
        16.5 metres of cable is more than a browser can draw, so every skeleton is decimated twice. The full detail
        keeps points within 2.0 µm of the original and no edge longer than 28 µm. The lite version, for machines without
        WebGPU, uses 8 µm and 150 µm and prunes terminal twigs shorter than 50 µm. Branch points, tips and the soma are
        always kept. Coordinates are in µm, and y is depth, with the pia (the brain&apos;s surface) at low y.
      </p>
      <p>Licence {credits.licence}. Please cite:</p>
      {credits.citations.map((c) => (
        <blockquote key={c} className="select-text border-l border-cyan-200/25 pl-4 text-slate-200">
          {c}
        </blockquote>
      ))}
      <p>
        More at{" "}
        <a href={credits.url} target="_blank" rel="noreferrer" className={LINK}>
          microns-explorer.org/cortical-mm3
        </a>
        .
      </p>
    </Section>
  );
}

const PARAMS: string[][] = [
  ["membrane time constant", `${DEFAULT_PARAMS.tauMs} ms`, "how fast a neuron forgets its input"],
  ["threshold", `${DEFAULT_PARAMS.vThreshold}`, `rest ${DEFAULT_PARAMS.vRest}, reset ${DEFAULT_PARAMS.vReset} (arbitrary units)`],
  ["refractory period", `${DEFAULT_PARAMS.refractoryMs} ms`, "after a spike, input is ignored"],
  ["excitatory weight", `+${DEFAULT_PARAMS.excWeight}`, "per synapse, before cleft scaling"],
  ["inhibitory weight", `${DEFAULT_PARAMS.inhWeight}`, "the sign comes from the presynaptic cell class"],
  ["cleft size gain", `${DEFAULT_PARAMS.sizeGain}`, "weight × (1 + gain × size / 255)"],
  ["conduction velocity", `${DEFAULT_PARAMS.conductionMps} m/s`, "along the axon, over each synapse's path distance"],
  ["slow motion", `${int.format(DEFAULT_PARAMS.slowMo)}×`, "applied to travel time only"],
  ["synaptic delay", `${DEFAULT_PARAMS.synDelayMs} ms`, "added once per synapse"],
  ["background drive", `${DEFAULT_PARAMS.backgroundRateHz} Hz`, `Poisson events per neuron, weight ${DEFAULT_PARAMS.backgroundWeight}`],
];

export function TheSimulation() {
  return (
    <Section id="simulation" kicker="03 · Dynamics" title="The simulation">
      <p>
        Every neuron is a leaky integrate-and-fire unit: input raises its voltage, the voltage leaks away, and when it
        crosses the threshold the cell spikes and resets. A spike travels down the cell&apos;s real axon to each of its
        real synapses, where it nudges the target up (excitatory) or down (inhibitory), by a weight scaled with the size
        of the synaptic cleft.
      </p>
      <p>
        A real spike crosses 800 µm of axon at about 0.5 m/s, in 1.6 ms, far too fast to see. Here that travel time is
        slowed 1,000 times, so the same pulse takes 1.6 s on screen. Membrane dynamics keep their real speed.
      </p>
      <Table caption="Default simulation parameters" head={["Parameter", "Value", "Note"]} rows={PARAMS} />
      <h3 className="pt-2 font-mono text-[11px] uppercase tracking-[0.24em] text-amber-200">Honest caveats</h3>
      <ul className="list-disc space-y-2 pl-5 marker:text-slate-600">
        <li>These are point neurons. There is no dendritic computation: a synapse far out on a branch counts the same as one at the soma.</li>
        <li>The data has no myelination, so one conduction velocity stands in for every axon.</li>
        <li>Nothing is fitted to recordings. The weights are round numbers, and the background drive is Poisson noise, there only to keep the volume alive.</li>
        <li>So this is a real wiring diagram with a toy dynamics on it. It shows the wiring, not a prediction of what the mouse&apos;s cortex does.</li>
      </ul>
    </Section>
  );
}

export function HowItWasBuilt() {
  return (
    <Section id="built" kicker="04 · Craft" title="How it was built">
      <p>
        Next.js 16 and React for the page and the HUD. The scene is three.js r186 on the WebGPURenderer, with the
        simulation running as TSL compute shaders, so every neuron and synapse steps on the GPU each frame. Without
        WebGPU it falls back to WebGL2 in a lite mode, with the same model running in a Web Worker. The sound is
        procedural Web Audio: nothing is sampled.
      </p>
      <p>
        Built with Claude Code by{" "}
        <a href="https://badcodes.dev" target="_blank" rel="noreferrer" className={LINK}>
          BadCodes
        </a>
        .
      </p>
    </Section>
  );
}

const CONTROLS: [string, string][] = [
  ["Click", "Select a cell and see its type, layer and connections"],
  ["Double-click", "Select and stimulate a cell"],
  ["Space", "Stimulate the selected cell and watch its signal travel down the axon"],
  ["R", "Ride the selected cell's next spike, from inside its axon"],
  ["Esc", "Cancel a ride, deselect, or free the mouse"],
  ["W A S D", "Fly forward, left, back, right"],
  ["Q / E", "Down and up"],
  ["Shift", "Fly faster"],
  ["Mouse", "Look around, once the mouse is captured by a click"],
];

export function Controls() {
  return (
    <Section id="controls" kicker="05 · Getting around" title="Controls">
      <ul className="divide-y divide-slate-500/15">
        {CONTROLS.map(([key, what]) => (
          <li key={key} className="flex flex-col gap-1 py-2 sm:flex-row sm:items-baseline sm:gap-4">
            <span className="sm:w-36 sm:shrink-0">
              <Kbd>{key}</Kbd>
            </span>
            <span>{what}</span>
          </li>
        ))}
      </ul>
      <p>On a touch screen, tap a cell; the left thumb moves and the right thumb looks.</p>
    </Section>
  );
}
