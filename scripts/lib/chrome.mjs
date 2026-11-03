/**
 * The one way the scripts launch a browser: the installed Google Chrome (Playwright, channel "chrome"),
 * never Playwright's bundled Chromium, whose build (156.0.8078.4 and older) is open to CVE-2026-118306
 * et al. Chrome follows stable, so every launch prints its version (a run that changes with a Chrome
 * release names it) and refuses a Chrome older than the first patched build.
 */

import { chromium } from "playwright";

export const PATCHED_CHROME = "156.0.8078.73";

const older = (version, floor) => {
  const a = version.split(".").map(Number);
  const b = floor.split(".").map(Number);
  for (let i = 0; i < b.length; i++) {
    if (!Number.isFinite(a[i])) return true; // a missing or non-numeric segment is refused
    if (a[i] !== b[i]) return a[i] < b[i];
  }
  return false;
};

/**
 * True when the launch failed because this machine cannot run the scripts' Chrome at all: Playwright's
 * "Chromium distribution 'chrome' is not found" or an unpatched Chrome. Any other launch failure (a
 * headed launch with no display, say) is a property of that launch, not of the machine.
 */
export const chromeUnusable = (err) => err?.chromeUnusable === true || /Chromium distribution '[^']+' is not found/.test(err?.message ?? "");

export async function launchChrome(options = {}) {
  const browser = await chromium.launch({ ...options, channel: "chrome" });
  try {
    const version = browser.version();
    console.log(`Chrome ${version}`);
    if (older(version, PATCHED_CHROME)) {
      const err = new Error(`Chrome ${version} is older than ${PATCHED_CHROME}, the first build patched for CVE-2026-118306 et al: update Chrome`);
      throw Object.assign(err, { chromeUnusable: true });
    }
    return browser;
  } catch (err) {
    // The caller never gets a browser it would have to close, so a refused or failed check closes it here.
    await browser.close();
    throw err;
  }
}
