import type { NextConfig } from "next";

/**
 * Sent on every route: no MIME sniffing, no full URLs to other sites (pinned rather than left to
 * each browser's default), and only the page itself and the badcodes.dev portfolio may frame it.
 * No X-Frame-Options goes with it: that header cannot name another origin, so SAMEORIGIN would
 * block the portfolio's embed in the very browsers that read it, and every browser that supports
 * frame-ancestors ignores it. Widening the list (previews, localhost) is a decision, not a fix.
 */
const SECURITY_HEADERS = [
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  { key: "Content-Security-Policy", value: "frame-ancestors 'self' https://badcodes.dev https://www.badcodes.dev" },
];

const nextConfig: NextConfig = {
  async headers() {
    return [{ source: "/:path*", headers: SECURITY_HEADERS }];
  },
};

export default nextConfig;
