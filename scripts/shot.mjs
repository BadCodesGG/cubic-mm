/**
 * Takes a reproducible screenshot of the hero shot from the production build.
 *
 *   npm run build && node scripts/shot.mjs [--synth] [--t=1.45] [--webgl] [--name=hero] [--port=3117] [--sim=cpu]
 *     [--quality=hi|lite|auto] [--pieces=1|2|3] [--mobile] [--live] [--frames=10] [--uncapped] [--gpu-timing] [--budgetMs=20]
 *
 * --hash=<c=...&n=...&v=...> opens a shareable link (implies nothing else: pair it with --live, since a
 * scripted shot keeps its own camera). --query=a=b&c=d adds URL parameters, e.g. --query=select=hero.
 * --clean hides the HUD, loader and sound button for the capture (the hero still is taken this way).
 * --type=<text> (with --live) presses "/" and types into the cell search before the capture; --enter
 * then presses Enter and waits for the flight. The camera view and the address-bar hash are printed.
 *
 * --hover=N|hero puts the pointer over cell N's soma (hero: a real mouse move onto the hero's soma, through the
 * picker) and waits for the wiring lines to fade in; --partners=N|hero selects that cell and pins its graph
 * (the "Show wiring" toggle), so the panel shows the partner counts. Both work with the scripted hero shot.
 *
 * --select starts with the hero selected; --stimulate=N (with --live) selects and stimulates neuron N, then waits --after=S s (default 4), or with --until-hop1=N until N cells have fired.
 * --quality / --pieces pin the render tier (default: the scripted shot's own, hi/2 on WebGPU).
 * --budgetMs lowers the adaptive frame budget (with --live), to watch the tier step down.
 * --mobile emulates a 390x844 phone (DPR 3, touch, coarse pointer, a phone user agent) and also
 * checks that the HUD panels do not overlap, the page has no horizontal overflow, the touch sticks
 * show, and the renderer's pixel ratio is at most 1.5. --live drops `shot=hero`, so the page runs
 * as a visitor sees it (free camera, adaptive quality) and is captured after 6 s.
 * --tour opens `/?tour=1&t=<--t>`: the intro tour on its fixed clock, held at t tour seconds.
 * Every other run marks the tour seen first, so it never covers the view being captured.
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
const mobile = flag("mobile");
const live = flag("live");
const tour = flag("tour");
const name = value("name", mobile ? "mobile" : "hero");
const query = new URLSearchParams(live || tour ? {} : { shot: "hero" });
if (tour) query.set("tour", "1");
if (flag("gpu-timing")) query.set("gpuTiming", "1");
for (const k of ["quality", "pieces", "budgetMs"]) {
  const v = value(k, null);
  if (v) query.set(k, v);
}
if (flag("synth")) query.set("synth", "1");
// --select: the hero starts selected, so the shot shows the cell panel (and the cascade panel under it).
if (flag("select")) query.set("select", "hero");
if (wantWebGL) query.set("webgl", "1");
const simMode = value("sim", null);
if (simMode) query.set("sim", simMode);
const t = value("t", null);
if (t) query.set("t", t);
for (const [k, v] of new URLSearchParams(value("query", ""))) query.set(k, v);
const linkHash = value("hash", "");
const url = `${BASE}/?${query}${linkHash ? `#${linkHash}` : ""}`;

const GPU_ARGS = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,UseSkiaRenderer", "--use-angle=d3d11", "--ignore-gpu-blocklist"];
// --uncapped: frames are not held to the display's refresh, so the frame time measures the GPU work.
if (flag("uncapped")) GPU_ARGS.push("--disable-gpu-vsync", "--disable-frame-rate-limit");
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

/** Runs in the page: overlap between HUD panels, horizontal overflow, touch sticks, pixel ratio. */
function layoutReport() {
  const boxes = [...document.querySelectorAll("[data-hud-panel]")]
    .map((el) => ({ name: el.getAttribute("data-hud-panel"), r: el.getBoundingClientRect() }))
    .filter((b) => b.r.width > 0 && b.r.height > 0);
  const overlaps = [];
  for (let i = 0; i < boxes.length; i++) {
    for (let j = i + 1; j < boxes.length; j++) {
      const a = boxes[i].r;
      const b = boxes[j].r;
      if (a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom) overlaps.push(`${boxes[i].name}/${boxes[j].name}`);
    }
  }
  const canvas = document.querySelector("canvas");
  return {
    panels: boxes.map((b) => b.name),
    overlaps,
    overflowX: document.documentElement.scrollWidth - window.innerWidth,
    sticks: document.querySelectorAll("[data-touch-stick]").length,
    pixelRatio: canvas ? canvas.width / canvas.clientWidth : 0,
  };
}

async function attempt(strategy) {
  const browser = await chromium.launch(strategy.launch);
  const errors = [];
  const threeWarnings = [];
  try {
    const context = await browser.newContext(
      mobile
        ? {
            viewport: { width: 390, height: 844 },
            deviceScaleFactor: 3,
            isMobile: true,
            hasTouch: true,
            userAgent:
              "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Mobile Safari/537.36",
          }
        : { viewport: { width: 1600, height: 900 }, deviceScaleFactor: 1 },
    );
    if (!tour) await context.addInitScript(() => localStorage.setItem("cmm-tour", "seen"));
    const page = await context.newPage();
    page.on("console", (msg) => {
      const text = msg.text();
      if (msg.type() === "error") errors.push(text);
      else if (text.includes("THREE.")) threeWarnings.push(text);
    });
    page.on("pageerror", (err) => errors.push(`pageerror: ${err.message}`));
    await page.goto(url, { waitUntil: "load" });
    if (live) {
      await page.waitForFunction(() => window.__cmm && window.__cmm.frame > 60, null, { timeout: 120_000 });
      await page.waitForTimeout(6000);
    } else {
      await page.waitForFunction(() => window.__cmm && window.__cmm.frame > 60 && window.__cmm.settled, null, {
        timeout: 120_000,
      });
    }
    // A few more frames so the held frame is fully resolved through bloom; --frames=N waits longer,
    // so the reported frame time is a steady-state mean rather than one that includes shader compiles.
    const extra = Number(value("frames", 10));
    await page.waitForFunction((f) => window.__cmm.frame > f, (await page.evaluate(() => window.__cmm.frame)) + extra, {
      timeout: 120_000,
    });
    const typed = value("type", null);
    if (typed !== null) {
      await page.keyboard.press("/");
      await page.keyboard.type(typed, { delay: 30 });
      await page.waitForTimeout(400);
      if (flag("enter")) {
        await page.keyboard.press("Enter");
        await page.waitForTimeout(6000);
      }
    }
    const viewNow = await page.evaluate(() => window.__cmm.view?.());
    const hashNow = await page.evaluate(() => location.hash);
    // --stimulate=N (with --live, whose clock keeps running): select and stimulate neuron N, then wait
    // --after=S seconds (default 4) so its cascade has spread before the frame is taken.
    const stimulateNeuron = value("stimulate", null);
    if (stimulateNeuron !== null) {
      await page.evaluate((n) => window.__cmm.stimulate(n), Number(stimulateNeuron));
      const untilHop1 = Number(value("until-hop1", 0));
      if (untilHop1 > 0) {
        // Take the frame as soon as N cells have fired, while their lines are still fresh.
        await page
          .waitForFunction((n) => window.__cmm.cascade().hop1 >= n, untilHop1, { timeout: Number(value("after", 8)) * 1000, polling: 50 })
          .catch(() => console.log(`  fewer than ${untilHop1} cells fired within --after`));
      } else {
        await page.waitForTimeout(Number(value("after", 4)) * 1000);
      }
    }
    const heroNeuron = () => page.evaluate(() => window.__cmm.hero.neuron);
    const hoverArg = value("hover", null);
    if (hoverArg === "hero") {
      const [x, y] = await page.evaluate(() => window.__cmm.hero.screen);
      await page.mouse.move(x - 60, y - 40);
      await page.mouse.move(x, y, { steps: 5 });
    } else if (hoverArg !== null) {
      await page.evaluate((n) => window.__cmm.wiring.hover(n), Number(hoverArg));
    }
    const partnersArg = value("partners", null);
    if (partnersArg !== null) {
      const n = partnersArg === "hero" ? await heroNeuron() : Number(partnersArg);
      await page.evaluate((k) => window.__cmm.wiring.pin(k), n);
    }
    // The graph fades in over 150 ms; give it, and the panel, time to settle.
    if (hoverArg !== null || partnersArg !== null) await page.waitForTimeout(900);
    const info = await page.evaluate(() => ({ ...window.__cmm }));
    info.wiringNow = await page.evaluate(() => {
      const w = window.__cmm.wiring;
      return w ? { buildMs: w.buildMs, maxPartners: w.maxPartners, lines: w.lines(), ...w.state() } : null;
    });
    Object.assign(info, { viewNow, hashNow });
    info.cascade = await page.evaluate(() => window.__cmm.cascade?.() ?? null);
    if (mobile) info.layout = await page.evaluate(layoutReport);
    if (!wantWebGL && !info.isWebGPU) return { ok: false, info, errors, threeWarnings, reason: "fell back to WebGL2" };
    if (flag("clean")) await page.addStyleTag({ content: "main > :not(canvas) { visibility: hidden !important; }" });
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
      `frame ${info.frame}, ${info.frameMs.toFixed(2)} ms/frame, ${info.gpuMs ? `GPU ${info.gpuMs.toFixed(2)} ms, ` : ""}${info.spikes} spikes, t=${info.simTime.toFixed(2)}s, ` +
      (info.sim ? `sim ${info.sim.mode}${info.sim.syntheticSynapses ? " (synthetic synapses)" : ""} on ${info.sim.synapses} synapses, ` : "") +
      `hero neuron ${info.hero.neuron} at ${info.hero.distanceUm.toFixed(0)} µm, on screen at ${info.hero.screen.map((v) => v.toFixed(0)).join(",")}`,
  );
  if (info.viewNow) console.log(`  view ${JSON.stringify(info.viewNow)}, selected ${info.selected}, hash ${info.hashNow || "(none)"}`);
  if (info.wiringNow) console.log(`  wiring ${JSON.stringify(info.wiringNow)}`);
  if (info.cascade) console.log(`  cascade ${JSON.stringify(info.cascade)}`);
  if (info.quality) {
    const q = info.quality;
    console.log(`  quality ${q.tier}/${q.pieces} (${q.instances} instances${q.adapted ? ", adapted down" : ""}), sim ${info.sim?.rateHz.toFixed(2)} Hz/neuron`);
  }
  const layoutProblems = [];
  if (info.layout) {
    const l = info.layout;
    console.log(`  layout: panels [${l.panels.join(", ")}], overlaps [${l.overlaps.join(", ")}], overflow-x ${l.overflowX}px, sticks ${l.sticks}, pixel ratio ${l.pixelRatio.toFixed(2)}`);
    if (l.overlaps.length) layoutProblems.push("HUD panels overlap");
    if (l.overflowX > 0) layoutProblems.push("horizontal overflow");
    if (l.sticks < 2) layoutProblems.push("touch sticks missing");
    if (l.pixelRatio > 1.5 + 1e-3) layoutProblems.push("pixel ratio above 1.5");
    if (info.quality?.tier !== "lite") layoutProblems.push("lite tier not picked");
  }
  console.log(`  saved ${path.relative(ROOT, result.out)}`);
  for (const p of layoutProblems) console.error(`  mobile check failed: ${p}`);
  if (layoutProblems.length) result.errors.push(...layoutProblems);
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
