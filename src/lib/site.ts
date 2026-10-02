import type { Metadata } from "next";

/**
 * Canonical origin for absolute URLs (metadataBase, social cards). Defaults to the production
 * domain so a build without the env var still emits absolute production URLs: a relative or
 * deployment URL is what makes Slack and Facebook drop the image. NEXT_PUBLIC_SITE_URL overrides it.
 */
export const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? "https://cubic.badcodes.dev";

export const SITE_NAME = "One Cubic Millimetre";
export const SITE_DESCRIPTION =
  "Walk through a real cubic millimetre of mouse brain: real neurons, real wiring, a live simulation.";

/**
 * The static share image, public/og.jpg. There is no opengraph-image file convention here, so the
 * root layout and every route's own metadata (see pageMetadata) must list it explicitly.
 */
export const OG_IMAGE = {
  url: "/og.jpg",
  width: 1200,
  height: 630,
  alt: "Real mouse-brain neurons and their wiring, glowing in the dark: One Cubic Millimetre",
};

/**
 * Metadata for a route that sets its own. Next replaces a layout's `openGraph` and `twitter`
 * wholesale when a route declares its own, so a route that set only its title would otherwise lose
 * the image or carry the layout's title.
 */
export function pageMetadata({
  title,
  description,
  path,
}: {
  title: string;
  description: string;
  path: string;
}): Metadata {
  return {
    title,
    description,
    alternates: { canonical: path },
    openGraph: { type: "website", siteName: SITE_NAME, title, description, url: path, images: [OG_IMAGE] },
    twitter: { card: "summary_large_image", title, description, images: [OG_IMAGE.url] },
  };
}
