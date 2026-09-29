/**
 * GPU/CPU parity for the spiking simulation, on the production build.
 *
 *   npm run build && node scripts/parity.mjs [--port=3121] [--synth]
 *
 * Loads `/?parity=1` twice in Playwright Chromium with WebGPU on: once with the simulation on the
 * GPU (TSL compute), once with `&sim=cpu` (the Web Worker). Both run the same seed on a
 * fixed 60 Hz clock for 5 simulated seconds and stimulate the hero neuron at t = 0.3 s.
 *
 * The two background RNGs differ (mulberry32 on the CPU, a PCG hash on the GPU), and the GPU
 * sums input in fixed point, so spike trains are compared statistically, not bit for bit:
 *   - total spikes within 15% of each other;
 *   - the hero's cascade reaches exactly the same set of first-hop post neurons.
 * Also fails on any console error or any `THREE.` warning.
 */

import { chromium } from "playwright";
import { startDevServer, stopDevServer, waitForServer } from "./lib/dev-server.mjs";

const argv = process.argv.slice(2);
const value = (name, fallback) => {
  const hit = argv.find((a) => a.startsWith(`--${name}=`));
  return hit ? hit.slice(name.length + 3) : fallback;
};
const PORT = Number(value("port", 3121));
const BASE = `http://localhost:${PORT}`;
const synth = argv.includes("--synth");
const TOLERANCE = 0.15;
const GPU_ARGS = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,UseSkiaRenderer", "--use-angle=d3d11", "--ignore-gpu-blocklist"];

async function run(browser, sim, benchSizes = []) {
  const query = new URLSearchParams({ parity: "1" });
  if (synth) query.set("synth", "1");
  if (sim) query.set("sim", sim);
  const context = await browser.newContext({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 1 });
  const page = await context.newPage();
  const problems = [];
  page.on("console", (msg) => {
    const text = msg.text();
    if (msg.type() === "error") problems.push(text);
    else if (text.includes("THREE.")) problems.push(`warning: ${text}`);
  });
  page.on("pageerror", (err) => problems.push(`pageerror: ${err.message}`));
  await page.goto(`${BASE}/?${query}`, { waitUntil: "load" });
  await page.waitForFunction(() => window.__cmm?.settled && window.__cmm.sim, null, { timeout: 180_000 });
  const result = await page.evaluate(async () => {
    const d = window.__cmm;
    await d.sim.settle();
    const counts = d.sim.spikeCounts();
    return {
      isWebGPU: d.isWebGPU,
      mode: d.sim.mode,
      syntheticSynapses: d.sim.syntheticSynapses,
      synapses: d.sim.synapses,
      dataset: d.dataset,
      simTime: d.simTime,
      frameMs: d.frameMs,
      spikes: d.sim.spikes,
      stimulated: d.sim.stimulated,
      rateHz: d.sim.rateHz,
      hero: d.hero.neuron,
      heroSpikes: counts[d.hero.neuron],
      neuronsThatFired: counts.filter((c) => c > 0).length,
      firstHop: d.sim.firstHop(),
    };
  });
  result.bench = [];
  for (const size of benchSizes) result.bench.push(await page.evaluate(([n, s]) => window.__cmm.sim.bench(n, s), size));
  await context.close();
  return { ...result, problems };
}

const server = startDevServer(PORT, "start");
let exitCode = 1;
try {
  await waitForServer(BASE);
  const browser = await chromium.launch({ headless: true, channel: "chromium", args: GPU_ARGS });
  try {
    // Per-frame cost at this dataset's size and at the full cube's (1,723 cells, 190k synapses).
    const gpu = await run(browser, null, [
      [200, 22_453],
      [1723, 190_000],
    ]);
    const cpu = await run(browser, "cpu");
    for (const [label, r] of [["GPU", gpu], ["CPU", cpu]]) {
      console.log(
        `${label}: sim ${r.mode} on ${r.dataset} data (${r.synapses} ${r.syntheticSynapses ? "synthetic " : ""}synapses), ` +
          `renderer ${r.isWebGPU ? "WebGPU" : "WebGL2"}, t=${r.simTime.toFixed(2)}s, ${r.frameMs.toFixed(2)} ms/frame, ` +
          `${r.spikes} spikes (${r.neuronsThatFired} neurons fired, ${r.rateHz.toFixed(2)} Hz/neuron over the last 2 s), ` +
          `hero ${r.hero} fired ${r.heroSpikes}x (${r.stimulated} by the stimulus), first hop reached ${r.firstHop.length} posts`,
      );
      for (const p of r.problems) console.log(`  ${label} console: ${p}`);
    }
    for (const b of gpu.bench) {
      const g = b.gpuMsPerFrame === null ? "n/a" : `${b.gpuMsPerFrame.toFixed(3)} ms`;
      console.log(
        `Cost per 60 Hz frame at ${b.neurons} neurons / ${b.synapses} synapses: GPU ${g}, ` +
          `CPU model ${b.cpuMsPerFrame.toFixed(3)} ms (${b.spikesPerSecond.toFixed(0)} spikes/s)`,
      );
    }
    const failures = [];
    if (gpu.mode !== "gpu") failures.push(`GPU run did not use GPU compute (mode ${gpu.mode}, WebGPU ${gpu.isWebGPU})`);
    if (cpu.mode !== "cpu") failures.push(`CPU run did not use the worker (mode ${cpu.mode})`);
    const diff = Math.abs(gpu.spikes - cpu.spikes) / Math.max(1, Math.max(gpu.spikes, cpu.spikes));
    console.log(`Total spikes: GPU ${gpu.spikes}, CPU ${cpu.spikes}, difference ${(diff * 100).toFixed(1)}% (limit ${TOLERANCE * 100}%)`);
    if (diff > TOLERANCE) failures.push(`spike totals differ by ${(diff * 100).toFixed(1)}%`);
    if (gpu.spikes === 0) failures.push("no spikes at all");
    for (const [label, r] of [["GPU", gpu], ["CPU", cpu]]) {
      if (r.stimulated !== 1) failures.push(`${label}: the stimulus caused ${r.stimulated} spikes, expected exactly 1`);
    }
    const g = new Set(gpu.firstHop);
    const c = new Set(cpu.firstHop);
    const onlyGpu = gpu.firstHop.filter((n) => !c.has(n));
    const onlyCpu = cpu.firstHop.filter((n) => !g.has(n));
    console.log(`First hop from hero: GPU ${g.size} posts, CPU ${c.size} posts, only-GPU [${onlyGpu}], only-CPU [${onlyCpu}]`);
    if (g.size === 0) failures.push("the hero's cascade reached no post neurons");
    if (onlyGpu.length || onlyCpu.length) failures.push("first-hop sets differ");
    if (gpu.problems.length || cpu.problems.length) failures.push("console errors or THREE warnings");
    if (failures.length) {
      console.error(`FAIL: ${failures.join("; ")}`);
    } else {
      console.log("PASS");
      exitCode = 0;
    }
  } finally {
    await browser.close();
  }
} catch (err) {
  console.error(err.message);
} finally {
  stopDevServer(server);
}
process.exit(exitCode);
