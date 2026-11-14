import { describe, expect, it } from "vitest";
import nextConfig from "../../next.config";

describe("security headers", () => {
  it("are sent on every route", async () => {
    const rules = (await nextConfig.headers?.()) ?? [];
    const all = rules.find((rule) => rule.source === "/:path*");
    expect(all).toBeDefined();
    const byKey = Object.fromEntries(all!.headers.map((h) => [h.key, h.value]));
    expect(byKey["X-Content-Type-Options"]).toBe("nosniff");
    expect(byKey["Referrer-Policy"]).toBe("strict-origin-when-cross-origin");
  });

  it("let only the page itself and the badcodes.dev portfolio frame it", async () => {
    const rules = (await nextConfig.headers?.()) ?? [];
    const csp = rules.flatMap((rule) => rule.headers).find((h) => h.key === "Content-Security-Policy")?.value ?? "";
    const ancestors = /frame-ancestors ([^;]+)/.exec(csp)?.[1].trim().split(/\s+/);
    expect(ancestors).toEqual(["'self'", "https://badcodes.dev", "https://www.badcodes.dev"]);
  });
});
