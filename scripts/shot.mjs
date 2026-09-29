/**
 * Takes a reproducible screenshot of the hero shot from the production build.
 *
 *   npm run build && node scripts/shot.mjs [--synth] [--t=1.45] [--webgl] [--name=hero] [--port=3117] [--sim=cpu]
 *
 * Starts `next start` on port 3117, opens `/?shot=hero` in Playwright Chromium at 1600x900,
 * DPR 1, waits for the frame counter and the shot clock to settle, asserts WebGPU (unless
 * --webgl) and zero console errors, and writes `.claude/shots/<name>-<n>.png` with the next
 * free n. Several launch strategies are tried in order, since WebGPU availability in
 * automated Chromium depends on headless mode and GPU flags; the one that worked is printed.
 */

import { mkdir, readdir } from "node:fs/promises";
import path from "node:path";
import { chromium } from "playwright";
import { ROOT, startDevServer, stopDevServer, waitForServer } from "./lib/dev-server.mjs";

const argv = process.argv.slice(2);
const flag = (name) => argv.includes(`--${name}`);
const value = (name, fallback) => {
  const hit = argv.find((a) => a.startsWith(`--${name}=`));
  return hit ? hit.slice(name.length + 3) : fallback;
};

// --port lets parallel worktrees each take their own.
const PORT = Number(value("port", 3117));
const BASE = `http://localhost:${PORT}`;
const SHOTS = path.join(ROOT, ".claude", "shots");

const wantWebGL = flag("webgl");
const name = value("name", "hero");
const query = new URLSearchParams({ shot: "hero" });
if (flag("synth")) query.set("synth", "1");
if (wantWebGL) query.set("webgl", "1");
const simMode = value("sim", null);
if (simMode) query.set("sim", simMode);
const t = value("t", null);
if (t) query.set("t", t);
const url = `${BASE}/?${query}`;

const GPU_ARGS = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,UseSkiaRenderer", "--use-angle=d3d11", "--ignore-gpu-blocklist"];
const STRATEGIES = [
  { label: "headless shell + GPU flags", launch: { headless: true, args: GPU_ARGS } },
  { label: "new headless (channel chromium) + GPU flags", launch: { headless: true, channel: "chromium", args: GPU_ARGS } },
  { label: "headed + GPU flags", launch: { headless: false, args: GPU_ARGS } },
  {
    label: "headless + SwiftShader WebGPU adapter",
    launch: { headless: true, args: [...GPU_ARGS, "--use-webgpu-adapter=swiftshader"] },
  },
];

async function nextShotPath() {
  await mkdir(SHOTS, { recursive: true });
  const taken = (await readdir(SHOTS))
    .map((f) => f.match(new RegExp(`^${name}-(\\d+)\\.png$`)))
    .filter(Boolean)
    .map((m) => Number(m[1]));
  const n = taken.length ? Math.max(...taken) + 1 : 1;
  return path.join(SHOTS, `${name}-${n}.png`);
}

async function attempt(strategy) {
  const browser = await chromium.launch(strategy.launch);
  const errors = [];
  const threeWarnings = [];
  try {
    const context = await browser.newContext({ viewport: { width: 1600, height: 900 }, deviceScaleFactor: 1 });
    const page = await context.newPage();
    page.on("console", (msg) => {
      const text = msg.text();
      if (msg.type() === "error") errors.push(text);
      else if (text.includes("THREE.")) threeWarnings.push(text);
    });
    page.on("pageerror", (err) => errors.push(`pageerror: ${err.message}`));
    await page.goto(url, { waitUntil: "load" });
    await page.waitForFunction(() => window.__cmm && window.__cmm.frame > 60 && window.__cmm.settled, null, {
      timeout: 120_000,
    });
    // A few more frames so the held frame is fully resolved through bloom.
    await page.waitForFunction((f) => window.__cmm.frame > f + 10, await page.evaluate(() => window.__cmm.frame));
    const info = await page.evaluate(() => ({ ...window.__cmm }));
    if (!wantWebGL && !info.isWebGPU) return { ok: false, info, errors, threeWarnings, reason: "fell back to WebGL2" };
    const out = await nextShotPath();
    await page.screenshot({ path: out });
    return { ok: true, info, errors, threeWarnings, out };
  } finally {
    await browser.close();
  }
}

const server = startDevServer(PORT, "start");
let exitCode = 1;
try {
  await waitForServer(BASE);
  let result = null;
  for (const strategy of STRATEGIES) {
    console.log(`Trying: ${strategy.label}`);
    try {
      result = await attempt(strategy);
    } catch (err) {
      console.log(`  failed: ${err.message.split("\n")[0]}`);
      continue;
    }
    if (result.ok) {
      console.log(`  worked: ${strategy.label}`);
      break;
    }
    console.log(`  rejected: ${result.reason}`);
  }
  if (!result?.ok) throw new Error("No launch strategy produced a WebGPU frame");
  for (const w of result.threeWarnings) console.log(`  THREE warning: ${w}`);
  const { info } = result;
  console.log(
    `  ${info.dataset} data, ${info.isWebGPU ? "WebGPU" : "WebGL2"}, ${info.instances} ribbon instances, ` +
      `frame ${info.frame}, ${info.frameMs.toFixed(2)} ms/frame, ${info.spikes} spikes, t=${info.simTime.toFixed(2)}s, ` +
      (info.sim ? `sim ${info.sim.mode}${info.sim.syntheticSynapses ? " (synthetic synapses)" : ""} on ${info.sim.synapses} synapses, ` : "") +
      `hero neuron ${info.hero.neuron} at ${info.hero.distanceUm.toFixed(0)} µm, on screen at ${info.hero.screen.map((v) => v.toFixed(0)).join(",")}`,
  );
  console.log(`  saved ${path.relative(ROOT, result.out)}`);
  if (result.errors.length) {
    console.error(`Console errors (${result.errors.length}):`);
    for (const e of result.errors) console.error(`  ${e}`);
  } else {
    exitCode = 0;
  }
} catch (err) {
  console.error(err.message);
} finally {
  stopDevServer(server);
}
process.exit(exitCode);
