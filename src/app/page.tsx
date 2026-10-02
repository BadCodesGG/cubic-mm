import type { Metadata } from "next";
import Experience from "@/components/experience";
import { pageMetadata, SITE_DESCRIPTION, SITE_NAME } from "@/lib/site";

export const metadata: Metadata = pageMetadata({ title: SITE_NAME, description: SITE_DESCRIPTION, path: "/" });

export default function Home() {
  return <Experience />;
}
