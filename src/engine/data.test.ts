import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { loadDataset } from "./data";
import { Compartment, NO_PARENT, type LodName } from "./format";

const DATA = fileURLToPath(new URL("../../public/data/", import.meta.url));

/** Serve `/data/...` straight from public/data, the way the static host would. */
function stubFetchFromDisk(requested: string[]) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      requested.push(url);
      if (!url.startsWith("/data/")) return new Response(null, { status: 404 });
      try {
        return new Response(new Uint8Array(readFileSync(DATA + url.slice("/data/".length))));
      } catch {
        return new Response(null, { status: 404 });
      }
    }),
  );
}

describe("loadDataset against the committed data", () => {
  const requested: string[] = [];
  beforeEach(() => {
    requested.length = 0;
    stubFetchFromDisk(requested);
  });
  afterEach(() => vi.unstubAllGlobals());

  for (const lod of ["hi", "lite"] as LodName[]) {
    it(`loads the ${lod} LOD into one consistent node index space`, async () => {
      const progress: string[] = [];
      const ds = await loadDataset("/data/", lod, (_l, _t, label) => progress.push(label));
      const { nodes, neurons, neuronNodeStart, neuronNodeCount, manifest } = ds;

      expect(ds.lod).toBe(lod);
      expect(ds.synapses).toBeNull();
      expect(neurons.count).toBe(manifest.neuronCount);
      expect(nodes.count).toBe(manifest.lods[lod].nodeCount);
      expect(nodes.pos.length).toBe(nodes.count * 3);
      expect(requested.filter((u) => u.includes("chunks/")).every((u) => u.includes(`chunks/${lod}/`))).toBe(true);
      expect(progress).toContain("manifest");

      // Neurons partition the node space in index order, each starting at its soma.
      let expectedStart = 0;
      for (let i = 0; i < neurons.count; i++) {
        expect(neuronNodeStart[i]).toBe(expectedStart);
        expect(neuronNodeCount[i]).toBeGreaterThan(0);
        expectedStart += neuronNodeCount[i];
        expect(nodes.compartment[neuronNodeStart[i]]).toBe(Compartment.Soma);
        expect(nodes.parent[neuronNodeStart[i]]).toBe(NO_PARENT);
      }
      expect(expectedStart).toBe(nodes.count);

      // neuronOfNode agrees with the ranges at every node, and no node is left unowned.
      for (let i = 0; i < neurons.count; i++) {
        const start = neuronNodeStart[i];
        const end = start + neuronNodeCount[i];
        if (nodes.neuronOfNode[start] !== i || nodes.neuronOfNode[end - 1] !== i) {
          throw new Error(`neuronOfNode disagrees with neuron ${i}'s range`);
        }
        for (let g = start + 1; g < end; g++) {
          if (nodes.neuronOfNode[g] !== i || nodes.parent[g] < start || nodes.parent[g] >= g) {
            throw new Error(`node ${g} is not a child inside neuron ${i}`);
          }
        }
      }
    });
  }

  it("rejects a missing manifest", async () => {
    await expect(loadDataset("/nope", "hi")).rejects.toThrow(/HTTP 404/);
  });
});
