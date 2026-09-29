/** Shared dev-server lifecycle for the scripts that need a running site. */

import { execFileSync, spawn } from "node:child_process";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");

export async function waitForServer(url, timeoutMs = 90_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(url, { redirect: "manual" });
      if (res.status < 500) return;
    } catch {
      // Not listening yet.
    }
    await new Promise((r) => setTimeout(r, 400));
  }
  throw new Error(`Timed out waiting for ${url} after ${timeoutMs}ms`);
}

/**
 * Invokes Next's CLI entrypoint through node rather than the .bin shim. The shim is a .cmd on
 * Windows, which spawn will only run under a shell -- and killing a shell orphans the server
 * it started, leaving the port bound for the next run.
 *
 * RESUME_SERVER=start runs the built output instead of dev. CI uses that so next/font serves
 * the fonts baked in at build time; a dev server re-resolving them can fall back to system
 * fonts, whose different metrics silently reflow the resume onto an extra page.
 */
export function startDevServer(port, mode = process.env.RESUME_SERVER ?? "dev") {
  const cli = path.join(ROOT, "node_modules", "next", "dist", "bin", "next");
  const child = spawn(process.execPath, [cli, mode, "--port", String(port)], {
    cwd: ROOT,
    detached: process.platform !== "win32",
    stdio: ["ignore", "pipe", "pipe"],
  });
  child.stdout.on("data", () => {});
  child.stderr.on("data", (d) => process.stderr.write(`  [next] ${d}`));
  return child;
}

/** Next spawns workers, so killing the parent alone can leave the port bound. */
export function stopDevServer(child) {
  if (!child?.pid) return;
  try {
    if (process.platform === "win32") {
      execFileSync("taskkill", ["/pid", String(child.pid), "/T", "/F"], { stdio: "ignore" });
    } else {
      process.kill(-child.pid, "SIGTERM");
    }
  } catch {
    child.kill("SIGKILL");
  }
}

/**
 * Next refuses to run two dev servers for the same directory, so scripts that boot their own
 * would fail whenever one is already up. It records the running server here; reusing it means
 * these checks work whether or not `npm run dev` is going.
 */
export async function findRunningServer() {
  try {
    const raw = await readFile(path.join(ROOT, ".next", "dev", "lock"), "utf8");
    const { appUrl } = JSON.parse(raw);
    if (!appUrl) return null;
    const res = await fetch(appUrl, { redirect: "manual" });
    return res.status < 500 ? appUrl : null;
  } catch {
    return null;
  }
}
