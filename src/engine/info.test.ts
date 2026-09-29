import { describe, expect, it } from "vitest";
import { tinyDataset } from "./camera/tiny-dataset";
import { CELL_TYPE_NAMES, datasetSummary, describeNeuron, formatCount, formatLength, piaDepth } from "./info";

describe("describeNeuron", () => {
  const ds = tinyDataset();

  it("names the cell type and class from the tables", () => {
    const a = describeNeuron(ds, 0);
    expect(a).toMatchObject({ typeCode: "23P", typeName: "layer 2/3 pyramidal", className: "excitatory", layer: 2 });
    expect(a.layerName).toBe("Layer 2");
    expect(a.rootId).toBe("864691135000000001");
    const b = describeNeuron(ds, 1);
    expect(b).toMatchObject({ typeCode: "BC", typeName: "basket cell", className: "inhibitory" });
  });

  it("sums edge lengths by compartment", () => {
    const a = describeNeuron(ds, 0);
    // axon edges 0-1 (10), 1-2 (20), 2-3 (30), 3-4 (30), 2-5 (20); dendrite edge 0-6 (20)
    expect(a.axonLengthUm).toBeCloseTo(10 + 20 + 30 + 30 + 20, 1);
    expect(a.dendriteLengthUm).toBeCloseTo(20, 1);
    expect(a.cableLengthUm).toBeCloseTo(a.axonLengthUm + a.dendriteLengthUm, 6);
    expect(a.nodeCount).toBe(7);
    const b = describeNeuron(ds, 1);
    expect(b.dendriteLengthUm).toBeCloseTo(100 + 70, 1);
    expect(b.axonLengthUm).toBeCloseTo(20, 1);
  });

  it("measures depth below the pia and reports the pia's provenance", () => {
    const pia = piaDepth(ds);
    expect(pia.source).toBe("layer-fit");
    expect(describeNeuron(ds, 0).depthBelowPiaUm).toBeCloseTo(Math.max(0, 200 - pia.y), 6);
  });

  it("counts synapses when the table exists and reports null when it does not", () => {
    const a = describeNeuron(ds, 0);
    expect([a.outgoing, a.incoming]).toEqual([1, 0]);
    const b = describeNeuron(ds, 1);
    expect([b.outgoing, b.incoming]).toEqual([0, 1]);
    const none = describeNeuron(tinyDataset({ synapses: false }), 0);
    expect([none.outgoing, none.incoming]).toEqual([null, null]);
  });

  it("rejects an index outside the dataset", () => {
    expect(() => describeNeuron(ds, 2)).toThrow(RangeError);
    expect(() => describeNeuron(ds, -1)).toThrow(RangeError);
  });

  it("carries the whole MICrONS cell-type map", () => {
    expect(Object.keys(CELL_TYPE_NAMES).sort()).toEqual(
      ["", "23P", "4P", "5P-ET", "5P-IT", "5P-NP", "6P-CT", "6P-IT", "BC", "BPC", "MC", "NGC"].sort(),
    );
    expect(CELL_TYPE_NAMES[""]).toBe("unclassified");
  });
});

describe("datasetSummary", () => {
  it("counts neurons, classes, types and cable", () => {
    const s = datasetSummary(tinyDataset());
    expect(s).toMatchObject({ neuronCount: 2, synapseCount: 1, excitatory: 1, inhibitory: 1, lod: "hi" });
    expect(s.perType.map((t) => t.code)).toEqual(["23P", "BC"]);
    expect(s.totalCableKm).toBeCloseTo((130 + 190) / 1e9, 9);
    expect(datasetSummary(tinyDataset({ synapses: false })).synapseCount).toBeNull();
  });
});

describe("formatting", () => {
  it("picks a readable unit", () => {
    expect(formatCount(190412)).toBe("190,412");
    expect(formatLength(340)).toBe("340 µm");
    expect(formatLength(1240)).toBe("1.24 mm");
    expect(formatLength(3.1e6)).toBe("3.10 m");
    expect(formatLength(2.5e9)).toBe("2.50 km");
  });
});
