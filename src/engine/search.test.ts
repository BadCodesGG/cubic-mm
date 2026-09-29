import { describe, expect, it } from "vitest";
import { nearestMatch, search, QUICK_QUERIES, type SearchData } from "./search";

const TYPES = ["", "23P", "4P", "5P-IT", "5P-ET", "BC", "MC", "BPC"];

// index: [type, layer, inhibitory, rootId, x]. y and z are 0, so the distance from the origin is |x|.
const ROWS: [number, number, number, string, number][] = [
  [1, 2, 0, "864691135000000001", 10], // 0: 23P, layer 2
  [2, 4, 0, "864691135000000002", 20], // 1: 4P
  [3, 5, 0, "864691136000000003", 30], // 2: 5P-IT
  [4, 5, 0, "864691136000000004", 40], // 3: 5P-ET
  [5, 4, 1, "864691137000000005", 50], // 4: BC
  [6, 2, 1, "864691137000000006", 60], // 5: MC
  [5, 5, 1, "864691137000000007", 5], // 6: BC, nearest of all
  [4, 5, 0, "864691136000000008", 70], // 7: 5P-ET
  [7, 3, 1, "864691137000000009", 80], // 8: BPC
  [0, 0, 0, "864691138000000010", 90], // 9: unclassified
];

const DATA: SearchData = {
  manifest: { cellTypes: TYPES },
  neurons: {
    count: ROWS.length,
    rootId: BigUint64Array.from(ROWS.map((r) => BigInt(r[3]))),
    somaUm: Float32Array.from(ROWS.flatMap((r) => [r[4], 0, 0])),
    cellType: Uint8Array.from(ROWS.map((r) => r[0])),
    inhibitory: Uint8Array.from(ROWS.map((r) => r[2])),
    layer: Uint8Array.from(ROWS.map((r) => r[1])),
  },
};

const ORIGIN = [0, 0, 0] as const;
const ids = (q: string, from: readonly [number, number, number] = ORIGIN) => search(DATA, q, from).hits.map((h) => h.neuron);

describe("search: root ids", () => {
  it("matches an exact root id", () => {
    expect(ids("864691136000000004")).toEqual([3]);
  });

  it("matches a prefix, nearest first", () => {
    expect(ids("8646911360")).toEqual([2, 3, 7]);
  });

  it("finds nothing for digits that start no id", () => {
    expect(ids("1234")).toEqual([]);
  });
});

describe("search: cell type codes", () => {
  it("matches a code in any case", () => {
    expect(ids("4P")).toEqual([1]);
    expect(ids("bc")).toEqual([6, 4]);
  });

  it("matches a hyphenated code with or without the hyphen", () => {
    expect(ids("5P-ET")).toEqual([3, 7]);
    expect(ids("5pet")).toEqual([3, 7]);
  });
});

describe("search: class words", () => {
  it("matches inhibitory and excitatory", () => {
    expect(ids("inhibitory")).toEqual([6, 4, 5, 8]);
    expect(ids("excitatory")).toEqual([0, 1, 2, 3, 7, 9]);
  });

  it("matches a cell-type name word: basket, martinotti, pyramidal", () => {
    expect(ids("basket")).toEqual([6, 4]);
    expect(ids("martinotti")).toEqual([5]);
    expect(ids("pyramidal")).toEqual([0, 1, 2, 3, 7]);
  });

  it("ignores articles and the word cell", () => {
    expect(ids("a basket cell")).toEqual([6, 4]);
    expect(ids("an inhibitory cell")).toEqual([6, 4, 5, 8]);
    expect(ids("a Martinotti cell")).toEqual([5]);
  });

  it("treats nearest cell as every cell", () => {
    expect(ids("nearest cell")).toHaveLength(8);
    expect(ids("nearest cell")[0]).toBe(6);
  });
});

describe("search: layers", () => {
  it("matches layer N, lN and layer N/M", () => {
    expect(ids("layer 4")).toEqual([1, 4]);
    expect(ids("l4")).toEqual([1, 4]);
    expect(ids("layer 2/3")).toEqual([0, 5, 8]);
  });

  it("combines with a subtype: layer 5 ET", () => {
    expect(ids("layer 5 ET")).toEqual([3, 7]);
    expect(ids("layer 5 IT")).toEqual([2]);
  });

  it("combines with a class: layer 5 inhibitory", () => {
    expect(ids("layer 5 inhibitory")).toEqual([6]);
  });
});

describe("search: results", () => {
  it("sorts by distance from the camera and reports it in µm", () => {
    const r = search(DATA, "inhibitory", [55, 0, 0]);
    expect(r.hits.map((h) => h.neuron)).toEqual([4, 5, 8, 6]);
    expect(r.hits.map((h) => h.distanceUm)).toEqual([5, 5, 25, 50]);
  });

  it("caps the list at eight and still counts every match", () => {
    const r = search(DATA, "nearest cell", ORIGIN);
    expect(r.hits).toHaveLength(8);
    expect(r.total).toBe(10);
  });

  it("describes each hit", () => {
    const [hit] = search(DATA, "864691137000000005", ORIGIN).hits;
    expect(hit).toEqual({
      neuron: 4,
      typeCode: "BC",
      typeName: "basket cell",
      layer: 4,
      rootId: "864691137000000005",
      distanceUm: 50,
    });
  });

  it("returns nothing for an empty or unknown query", () => {
    expect(search(DATA, "", ORIGIN)).toEqual({ hits: [], total: 0 });
    expect(search(DATA, "   ", ORIGIN)).toEqual({ hits: [], total: 0 });
    expect(search(DATA, "zebrafish", ORIGIN)).toEqual({ hits: [], total: 0 });
    expect(search(DATA, "basket zebrafish", ORIGIN).total).toBe(0);
    expect(search(DATA, "layer", ORIGIN).total).toBe(0);
    expect(search(DATA, "layer 9", ORIGIN).total).toBe(0);
  });
});

describe("nearestMatch", () => {
  it("is the nearest hit", () => {
    expect(nearestMatch(DATA, "a basket cell", ORIGIN, -1)?.neuron).toBe(6);
  });

  it("skips the cell that is already selected, so a chip moves on", () => {
    expect(nearestMatch(DATA, "a basket cell", ORIGIN, 6)?.neuron).toBe(4);
  });

  it("falls back to the selected cell when it is the only match", () => {
    expect(nearestMatch(DATA, "a Martinotti cell", ORIGIN, 5)?.neuron).toBe(5);
  });

  it("is null when nothing matches", () => {
    expect(nearestMatch(DATA, "zebrafish", ORIGIN, -1)).toBeNull();
  });
});

describe("QUICK_QUERIES", () => {
  it("are the five chips, and each one parses to a query that finds a cell", () => {
    expect(QUICK_QUERIES).toEqual([
      "a basket cell",
      "a Martinotti cell",
      "layer 5 ET",
      "an inhibitory cell",
      "nearest cell",
    ]);
    for (const q of QUICK_QUERIES) expect(search(DATA, q, ORIGIN).total).toBeGreaterThan(0);
  });
});
