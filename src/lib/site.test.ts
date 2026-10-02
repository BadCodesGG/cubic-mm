import { describe, expect, it, vi } from "vitest";
// next/font/google only exists inside the Next build; the layout needs it just to be importable.
vi.mock("next/font/google", () => ({ Geist: () => ({ variable: "" }), Geist_Mono: () => ({ variable: "" }) }));

import { metadata as layoutMetadata } from "@/app/layout";
import { metadata as aboutMetadata } from "@/app/about/page";
import { OG_IMAGE, pageMetadata, shareTitle, SITE_NAME, SITE_URL } from "@/lib/site";

type Images = unknown[] | undefined;
const ogImages = (m: { openGraph?: unknown }) => (m.openGraph as { images?: Images }).images;
const twitterImages = (m: { twitter?: unknown }) => (m.twitter as { images?: Images }).images;

describe("share image", () => {
  it("SITE_URL is the https production domain", () => {
    expect(SITE_URL).toBe("https://cubic.badcodes.dev");
  });

  it("OG_IMAGE points at the static jpeg", () => {
    expect(OG_IMAGE).toMatchObject({ url: "/og.jpg", width: 1200, height: 630 });
  });

  it("pageMetadata carries the image and its own title, description and url", () => {
    const m = pageMetadata({ title: "T", description: "D", path: "/x" });
    expect(ogImages(m)).toContain(OG_IMAGE);
    expect(twitterImages(m)).toContain(OG_IMAGE.url);
    expect(m.openGraph).toMatchObject({ title: "T | One Cubic Millimetre", description: "D", url: "/x" });
    expect(m.alternates?.canonical).toBe("/x");
  });
  it("a card's title is the page's <title>: the root page as written, every other route with the suffix", () => {
    expect(shareTitle("T", "/")).toBe("T");
    expect(shareTitle("T", "/x")).toBe(`T | ${SITE_NAME}`);
  });

  it("/about's card says which site it is, as its <title> does", () => {
    expect(aboutMetadata.openGraph).toMatchObject({ title: "About | One Cubic Millimetre", url: "/about" });
    expect(aboutMetadata.twitter).toMatchObject({ title: "About | One Cubic Millimetre" });
  });

  it.each([
    ["layout", layoutMetadata],
    ["/about", aboutMetadata],
  ])("%s lists the og image for openGraph and twitter", (_name, m) => {
    expect(ogImages(m)).toContain(OG_IMAGE);
    expect(twitterImages(m)).toContain(OG_IMAGE.url);
  });

  it("/about has its own og:url", () => {
    expect((aboutMetadata.openGraph as { url?: string }).url).toBe("/about");
  });
});
