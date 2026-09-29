/** Canonical origin for absolute URLs (metadataBase, social cards). Set NEXT_PUBLIC_SITE_URL on a deployment. */
export const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000";

export const SITE_NAME = "One Cubic Millimetre";
export const SITE_DESCRIPTION =
  "Walk through a real cubic millimetre of mouse brain: real neurons, real wiring, a live simulation.";

/**
 * The generated card at src/app/opengraph-image.tsx. Next merges that file into a route's Open
 * Graph tags only when the route declares no `openGraph` block of its own, so any page that sets
 * one (like /about) must restate this or it ships a card with no picture.
 */
export const OG_IMAGE = { url: "/opengraph-image", width: 1200, height: 630, alt: SITE_NAME };
