import { readFile } from "node:fs/promises";
import path from "node:path";
import { ImageResponse } from "next/og";
import { datasetFacts } from "@/lib/dataset-facts";
import { OG_IMAGE, SITE_NAME } from "@/lib/site";

export const alt = SITE_NAME;
export const size = { width: OG_IMAGE.width, height: OG_IMAGE.height };
export const contentType = "image/png";

const int = new Intl.NumberFormat("en-US");

export default async function OpengraphImage() {
  const facts = datasetFacts();
  const still = await readFile(path.join(process.cwd(), "public", "hero-still.jpg"), "base64");
  return new ImageResponse(
    (
      <div style={{ width: "100%", height: "100%", display: "flex", position: "relative", background: "#04060b" }}>
        {/* eslint-disable-next-line @next/next/no-img-element -- satori renders this, not the browser */}
        <img src={`data:image/jpeg;base64,${still}`} alt="" width={1200} height={630} style={{ position: "absolute", inset: 0, objectFit: "cover" }} />
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            background: "linear-gradient(to top, rgba(4,6,11,0.92) 0%, rgba(4,6,11,0.55) 45%, rgba(4,6,11,0.1) 100%)",
          }}
        />
        <div style={{ position: "relative", display: "flex", flexDirection: "column", justifyContent: "flex-end", padding: "64px", width: "100%" }}>
          <div style={{ fontSize: 76, fontWeight: 700, color: "#f1f5f9", lineHeight: 1.05 }}>{SITE_NAME}</div>
          <div style={{ fontSize: 34, color: "#a5f3fc", marginTop: 20 }}>
            {`${int.format(facts.neuronCount)} real neurons · ${int.format(facts.synapseCount)} synapses`}
          </div>
        </div>
      </div>
    ),
    { ...size },
  );
}
