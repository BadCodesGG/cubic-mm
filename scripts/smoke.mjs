/**
 * End-to-end smoke test on the production build.
 *
 *   npm run build && npm run test:smoke [-- --port=3132] [--allow-software] [--skip-parity]
 *
 * Boots `next start` (it refuses to run without a build: dev mode hides prerender problems) and
 * drives it in Playwright Chromium, twice:
 *   a. WebGPU on: the GPU simulation runs, a click + Space stimulates a cell, R starts a ride;
 *      /about carries the citation and the synapse count; the social image is a PNG.
 *   b. WebGPU off (`?webgl=1`): the WebGL2 lite mode runs the CPU simulation.
 * then runs scripts/parity.mjs on the same port. Prints a line per assertion, exits 1 on the first failure.
 *
 * --allow-software is for CI runners, which have no GPU: WebGPU may fall back to SwiftShader or be
 * absent, so the `isWebGPU === true` and `mode === "gpu"` assertions and the GPU/CPU parity run are
 * skipped, but zero console errors and a running, stimulable simulation are still required.
 */

import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { chromium } from "playwright";
import { ROOT, startDevServer, stopDevServer, waitForServer } from "./lib/dev-server.mjs";

const argv = process.argv.slice(2);
const flag = (name) => argv.includes(`--${name}`);
const value = (name, fallback) => {
  const hit = argv.find((a) => a.startsWith(`--${name}=`));
  return hit ? hit.slice(name.length + 3) : fallback;
};
const PORT = Number(value("port", 3132));
const BASE = `http://localhost:${PORT}`;
const SOFTWARE = flag("allow-software");

const GPU_ARGS = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,UseSkiaRenderer", "--use-angle=d3d11", "--ignore-gpu-blocklist"];
// A GPU-less runner gets no WebGPU flags: the page then takes whatever fallback the browser really offers
// (a SwiftShader WebGPU adapter stalls the engine for minutes, and is not what a visitor without a GPU gets).
const WEBGPU_ARGS = SOFTWARE ? ["--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] : GPU_ARGS;
// --disable-webgpu is best effort (Chromium ignores switches it does not know); `?webgl=1` is what forces the fallback.
const WEBGL_ARGS = ["--disable-webgpu", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"];

function fail(message) {
  console.error(`FAIL  ${message}`);
  throw new Error(message);
}

function check(label, ok, detail = "") {
  if (!ok) fail(`${label}${detail ? ` (${detail})` : ""}`);
  console.log(`PASS  ${label}${detail ? ` (${detail})` : ""}`);
}

function skip(label, why) {
  console.log(`SKIP  ${label} (${why})`);
}

/** Console problems: any error, any pageerror, any THREE.* warning. */
function watch(page) {
  const problems = [];
  page.on("console", (msg) => {
    const text = msg.text();
    if (msg.type() === "error") problems.push(text);
    else if (text.includes("THREE.")) problems.push(`warning: ${text}`);
  });
  page.on("pageerror", (err) => problems.push(`pageerror: ${err.message}`));
  return problems;
}

async function launch(args) {
  return chromium.launch({ headless: true, channel: "chromium", args });
}

async function openExperience(browser, query, viewport = { width: 1280, height: 720 }) {
  const context = await browser.newContext({ viewport, deviceScaleFactor: 1 });
  const page = await context.newPage();
  const problems = watch(page);
  await page.goto(`${BASE}/?${query}`, { waitUntil: "load" });
  await page.waitForFunction(() => window.__cmm && window.__cmm.frame > 30 && window.__cmm.sim, null, { timeout: 180_000 });
  return { context, page, problems };
}

const html = async (route) => {
  const res = await fetch(`${BASE}${route}`);
  return { res, text: await res.text() };
};

const decode = (s) => s.replace(/&#x27;/g, "'").replace(/&quot;/g, '"').replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");

async function runWebGPU(browser) {
  console.log(`-- a. WebGPU run${SOFTWARE ? " (software allowed)" : ""}`);
  const { context, page, problems } = await openExperience(browser, "select=hero");
  try {
    const info = await page.evaluate(() => ({
      isWebGPU: window.__cmm.isWebGPU,
      mode: window.__cmm.sim.mode,
      dataset: window.__cmm.dataset,
      selected: window.__cmm.selected,
      hero: window.__cmm.hero.screen,
    }));
    if (SOFTWARE) skip("__cmm.isWebGPU === true", `--allow-software, got ${info.isWebGPU}`);
    else check("__cmm.isWebGPU === true", info.isWebGPU === true, String(info.isWebGPU));
    if (SOFTWARE) skip('__cmm.sim.mode === "gpu"', `--allow-software, got ${info.mode}`);
    else check('__cmm.sim.mode === "gpu"', info.mode === "gpu", info.mode);
    check('__cmm.dataset === "real"', info.dataset === "real", info.dataset);
    check("?select=hero selected the hero neuron", info.selected >= 0, `neuron ${info.selected}`);

    // Click the canvas centre (the visitor's gesture: it also asks for pointer lock), then make sure a
    // cell is selected: a click that lands on empty space deselects, so re-pick the hero if that happened.
    const size = page.viewportSize();
    await page.mouse.click(size.width / 2, size.height / 2);
    if ((await page.evaluate(() => window.__cmm.selected)) < 0) {
      const [hx, hy] = await page.evaluate(() => window.__cmm.hero.screen);
      await page.mouse.click(hx, hy);
    }
    await page.waitForFunction(() => window.__cmm.selected >= 0, null, { timeout: 5000 }).catch(() => {});
    const selected = await page.evaluate(() => window.__cmm.selected);
    check("a cell is selected after clicking the canvas", selected >= 0, `neuron ${selected}`);

    const before = await page.evaluate(() => window.__cmm.sim.spikes);
    await page.keyboard.press("Space");
    await page.waitForTimeout(2000);
    await page.evaluate(() => window.__cmm.sim.settle());
    await page.waitForFunction(() => window.__cmm.sim.stimulated >= 1, null, { timeout: 15_000 }).catch(() => {});
    const after = await page.evaluate(() => ({ stimulated: window.__cmm.sim.stimulated, spikes: window.__cmm.sim.spikes }));
    check("Space: __cmm.sim.stimulated >= 1", after.stimulated >= 1, String(after.stimulated));
    check("__cmm.sim.spikes increased", after.spikes > before, `${before} -> ${after.spikes}`);

    await page.keyboard.press("r");
    const ride = await page
      .waitForFunction(() => window.__cmm.ride !== "idle" && window.__cmm.ride, null, { timeout: 5000 })
      .then((h) => h.jsonValue())
      .catch(() => "idle");
    check('R: __cmm.ride leaves "idle" within 5 s', ride !== "idle", String(ride));

    check("no console errors or THREE warnings", problems.length === 0, problems.join(" | "));
  } finally {
    await context.close();
  }

  const manifest = JSON.parse(await readFile(path.join(ROOT, "public", "data", "manifest.json"), "utf8"));
  const citation = manifest.credits.citations[0];
  const synapses = new Intl.NumberFormat("en-US").format(manifest.synapses.count);
  const about = await html("/about");
  check("/about returns 200", about.res.status === 200, String(about.res.status));
  const aboutText = decode(about.text);
  check("/about contains the citation", aboutText.includes(citation));
  check(`/about contains the synapse count (${synapses})`, aboutText.includes(synapses));
  const ogMatch = about.text.match(/<meta property="og:image" content="([^"]*)"/);
  check("/about restates og:image", !!ogMatch, ogMatch?.[1] ?? "missing");
  const home = await html("/");
  check("/ has og:image", /<meta property="og:image" content="/.test(home.text));
  // metadataBase is the deployment origin, not this server, so fetch the path from here.
  const og = await fetch(`${BASE}${new URL(ogMatch[1]).pathname}`);
  const type = og.headers.get("content-type") ?? "";
  check("OG image returns 200 image/png", og.status === 200 && type.startsWith("image/png"), `${og.status} ${type}`);
}

async function runWebGL(browser) {
  console.log("-- b. WebGPU disabled (?webgl=1)");
  const { context, page, problems } = await openExperience(browser, "webgl=1");
  try {
    const info = await page.evaluate(() => ({ isWebGPU: window.__cmm.isWebGPU, mode: window.__cmm.sim.mode }));
    check("__cmm.isWebGPU === false", info.isWebGPU === false, String(info.isWebGPU));
    check('__cmm.sim.mode === "cpu"', info.mode === "cpu", info.mode);
    await page.waitForFunction(() => window.__cmm.sim.spikes > 0, null, { timeout: 30_000 }).catch(() => {});
    const spikes = await page.evaluate(() => window.__cmm.sim.spikes);
    check("the CPU simulation spiked", spikes > 0, `${spikes} spikes`);
    check("no console errors or THREE warnings", problems.length === 0, problems.join(" | "));
  } finally {
    await context.close();
  }

  // /about must be readable at phone width: no sideways scroll.
  const phone = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2 });
  try {
    const page = await phone.newPage();
    const problems = watch(page);
    await page.goto(`${BASE}/about`, { waitUntil: "load" });
    const { scrollWidth, innerWidth } = await page.evaluate(() => ({
      scrollWidth: document.documentElement.scrollWidth,
      innerWidth: window.innerWidth,
    }));
    check("/about does not scroll sideways at 390 px", scrollWidth <= innerWidth, `${scrollWidth} <= ${innerWidth}`);
    check("/about has no console errors", problems.length === 0, problems.join(" | "));
  } finally {
    await phone.close();
  }
}

if (!existsSync(path.join(ROOT, ".next", "BUILD_ID"))) {
  console.error("FAIL  no production build: run `npm run build` first (the smoke test serves `next start`)");
  process.exit(1);
}

let exitCode = 1;
let server = startDevServer(PORT, "start");
try {
  await waitForServer(BASE);
  console.log(`serving the production build on ${BASE}`);
  const gpu = await launch(WEBGPU_ARGS);
  try {
    await runWebGPU(gpu);
  } finally {
    await gpu.close();
  }
  const soft = await launch(WEBGL_ARGS);
  try {
    await runWebGL(soft);
  } finally {
    await soft.close();
  }

  // parity.mjs boots its own server on the port it is given, so free ours first.
  stopDevServer(server);
  server = null;
  console.log("-- c. GPU/CPU parity (scripts/parity.mjs)");
  if (SOFTWARE) {
    skip("parity", "--allow-software: it needs real WebGPU compute");
  } else if (flag("skip-parity")) {
    skip("parity", "--skip-parity");
  } else {
    const parity = spawnSync(process.execPath, [path.join(ROOT, "scripts", "parity.mjs"), `--port=${PORT}`], {
      cwd: ROOT,
      encoding: "utf8",
      env: { ...process.env },
    });
    const lines = `${parity.stdout}${parity.stderr}`.trim().split("\n");
    for (const l of lines) console.log(`      ${l}`);
    check("parity.mjs PASS", parity.status === 0 && lines.some((l) => l.trim() === "PASS"), `exit ${parity.status}`);
  }
  console.log("SMOKE OK");
  exitCode = 0;
} catch (err) {
  console.error(`smoke failed: ${err.message}`);
} finally {
  if (server) stopDevServer(server);
}
process.exit(exitCode);
