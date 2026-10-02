import type { Metadata } from "next";
import Link from "next/link";
import { datasetFacts } from "@/lib/dataset-facts";
import { pageMetadata, SITE_NAME } from "@/lib/site";
import { Controls, HowItWasBuilt, TheData, TheSimulation, WhatYouAreLookingAt } from "@/components/about/sections";
import { LINK } from "@/components/about/section";

const int = new Intl.NumberFormat("en-US");
const facts = datasetFacts();
const description = `${int.format(facts.neuronCount)} real neurons and ${int.format(facts.synapseCount)} real synapses from the MICrONS cubic millimetre of mouse cortex, with a toy spiking simulation on the true wiring.`;

export const metadata: Metadata = pageMetadata({ title: "About", description, path: "/about" });

export default function About() {
  return (
    <main className="mx-auto w-full max-w-3xl px-5 pb-16 pt-8 sm:px-8">
      <nav aria-label="Back" className="font-mono text-[11px] uppercase tracking-[0.2em]">
        <Link href="/" className={LINK}>
          ← Back to the volume
        </Link>
      </nav>
      <header className="pb-10 pt-10">
        <p className="font-mono text-[11px] uppercase tracking-[0.28em] text-slate-400">{SITE_NAME}</p>
        <h1 className="mt-3 text-4xl font-semibold tracking-tight text-slate-50 sm:text-5xl">
          A real piece of brain, and the parts that are invented
        </h1>
        <p className="mt-4 text-lg leading-relaxed text-slate-300">
          The neurons and their wiring are measured. The activity running on them is a toy. This page says which is which.
        </p>
      </header>
      <WhatYouAreLookingAt facts={facts} />
      <TheData facts={facts} />
      <TheSimulation />
      <HowItWasBuilt />
      <Controls />
      <footer className="border-t border-slate-500/20 pt-8">
        <Link href="/" className={`${LINK} font-mono text-[11px] uppercase tracking-[0.2em]`}>
          ← Back to the volume
        </Link>
      </footer>
    </main>
  );
}
