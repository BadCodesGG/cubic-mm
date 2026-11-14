import type { NextConfig } from "next";

/** Sent on every route: no MIME sniffing, no full URLs to other sites, and only the portfolio (badcodes.dev) may frame the pages. */
export const SECURITY_HEADERS = [
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
