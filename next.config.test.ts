import { describe, expect, it } from "vitest";
import nextConfig from "./next.config";

describe("security headers", () => {
  // Next applies every matching rule and a later rule's value for the same key wins, so a second
  // rule could quietly override these for some paths. One catch-all rule is the whole contract.
  const catchAll = async () => {
    const rules = (await nextConfig.headers?.()) ?? [];
    expect(rules.map((rule) => rule.source)).toEqual(["/:path*"]);
    expect(rules[0].has ?? rules[0].missing).toBeUndefined();
    return Object.fromEntries(rules[0].headers.map((h) => [h.key, h.value]));
  };

  it("are sent on every route by the one catch-all rule", async () => {
    const byKey = await catchAll();
    expect(byKey["X-Content-Type-Options"]).toBe("nosniff");
    expect(byKey["Referrer-Policy"]).toBe("strict-origin-when-cross-origin");
    expect(byKey["Content-Security-Policy"]).toMatch(/^frame-ancestors [^;]+$/);
  });

  it("let only the page itself and the badcodes.dev portfolio frame it", async () => {
    const csp = (await catchAll())["Content-Security-Policy"] ?? "";
    const ancestors = csp.replace(/^frame-ancestors\s+/, "").trim().split(/\s+/).sort();
    expect(ancestors).toEqual(["'self'", "https://badcodes.dev", "https://www.badcodes.dev"]);
  });
});
