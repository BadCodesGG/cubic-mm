import { beforeEach, describe, expect, it, vi } from "vitest";

const { launch } = vi.hoisted(() => ({ launch: vi.fn() }));
vi.mock("playwright", () => ({ chromium: { launch } }));

import { chromeUnusable, launchChrome } from "./chrome.mjs";

function fakeBrowser(version: () => string) {
  const browser = { version, close: vi.fn(async () => {}) };
  launch.mockResolvedValueOnce(browser);
  return browser;
}

describe("launchChrome", () => {
  beforeEach(() => {
    launch.mockReset();
    vi.spyOn(console, "log").mockImplementation(() => {});
  });

  it.each(["156.0.8078.73", "156.0.8078.80", "157.0.1.0"])("returns the open browser for Chrome %s", async (version) => {
    const browser = fakeBrowser(() => version);
    await expect(launchChrome()).resolves.toBe(browser);
    expect(browser.close).not.toHaveBeenCalled();
  });

  it.each(["156.0.8078.4", "155.9.9999.99", "156.0", "", "156.0.8078.73 beta", "abc"])(
    "refuses Chrome %j as unusable and closes it",
    async (version) => {
      const browser = fakeBrowser(() => version);
      const err = await launchChrome().catch((e: unknown) => e);
      expect(err).toBeInstanceOf(Error);
      expect(chromeUnusable(err)).toBe(true);
      expect(browser.close).toHaveBeenCalledOnce();
    },
  );

  it("closes the browser when reading its version throws", async () => {
    const browser = fakeBrowser(() => {
      throw new Error("boom");
    });
    await expect(launchChrome()).rejects.toThrow("boom");
    expect(browser.close).toHaveBeenCalledOnce();
  });

  it("always launches the installed Chrome, whatever channel is passed", async () => {
    fakeBrowser(() => "156.0.8078.80");
    await launchChrome({ headless: true, channel: "chromium" });
    expect(launch).toHaveBeenCalledWith({ headless: true, channel: "chrome" });
  });
});
