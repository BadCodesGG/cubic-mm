/**
 * Cell search: plain matching over the neuron table, no fuzzy library. Pure and unit-testable.
 *
 * A query is one of:
 *   digits            a root id, exact or prefix ("864691136")
 *   words             every word must hold (AND), articles and "cell" are ignored:
 *     4P, 5P-ET, 5pet   a cell-type code, any case, hyphen optional
 *     et, it, np, ct    a layer 5 or 6 subtype: matches 5P-ET, 6P-IT, ...
 *     basket, martinotti, pyramidal, bipolar, neurogliaform
 *                       a word of the type's name (three letters or more, prefix match)
 *     inhibitory, excitatory   the class (three letters or more, prefix match)
 *     layer 5, l5, layer 2/3   the cortical layer
 *     nearest, any      no filter: every cell
 * Results are sorted by distance from the camera, nearest first.
 */

import type { Manifest, NeuronTable } from "./format";
import { cellTypeName } from "./info";

export interface SearchData {
  manifest: Pick<Manifest, "cellTypes">;
  neurons: NeuronTable;
}

export interface SearchHit {
  neuron: number;
  typeCode: string;
  typeName: string;
  /** Cortical layer 1 to 6, 0 if unknown. */
  layer: number;
  rootId: string;
  /** Straight-line distance from the camera to the soma, µm. */
  distanceUm: number;
}

export interface SearchResult {
  hits: SearchHit[];
  /** Every match, not only the ones listed. */
  total: number;
}

export const MAX_RESULTS = 8;

/** The quick chips under the search box. Each is a query as well as a label. */
export const QUICK_QUERIES: readonly string[] = ["a basket cell", "a Martinotti cell", "layer 5 ET", "an inhibitory cell", "nearest cell"];

const NOISE = new Set(["a", "an", "the", "cell", "cells", "neuron", "neurons", "nearest", "closest", "any"]);
const LAYER_NUMBER = /^[1-6](\/[1-6])?$/;
const SUBTYPES = new Set(["it", "et", "np", "ct"]);

type Predicate = (neuron: number) => boolean;

const normaliseCode = (s: string) => s.toUpperCase().replace(/-/g, "");

/** The predicate a query means, or null when it means nothing (empty, or a word this grammar does not know). */
function compile(data: SearchData, query: string): Predicate | null {
  const q = query.trim().toLowerCase().replace(/\s+/g, " ");
  if (!q) return null;
  const { neurons, manifest } = data;

  if (/^\d+$/.test(q)) return (i) => neurons.rootId[i].toString().startsWith(q);

  const codes = manifest.cellTypes;
  const typeSet = (test: (code: string) => boolean): Set<number> => {
    const set = new Set<number>();
    codes.forEach((code, index) => {
      if (test(code)) set.add(index);
    });
    return set;
  };

  const preds: Predicate[] = [];
  const tokens = q.split(" ");
  for (let t = 0; t < tokens.length; t++) {
    const token = tokens[t];
    if (NOISE.has(token)) continue;

    let layers: number[] | null = null;
    if (token === "layer" || token === "layers") {
      const next = tokens[++t];
      if (!next || !LAYER_NUMBER.test(next)) return null;
      layers = next.split("/").map(Number);
    } else if (/^l[1-6]$/.test(token)) {
      layers = [Number(token[1])];
    }
    if (layers) {
      const wanted = layers;
      preds.push((i) => wanted.includes(neurons.layer[i]));
      continue;
    }

    if (token.length >= 3 && "inhibitory".startsWith(token)) {
      preds.push((i) => neurons.inhibitory[i] === 1);
      continue;
    }
    if (token.length >= 3 && "excitatory".startsWith(token)) {
      preds.push((i) => neurons.inhibitory[i] === 0);
      continue;
    }

    // A word this dataset has no cell for finds nothing, the same as a word the grammar does not know.
    let types = typeSet((code) => code !== "" && normaliseCode(code) === normaliseCode(token));
    if (types.size === 0 && SUBTYPES.has(token)) types = typeSet((code) => code.toLowerCase().endsWith(`-${token}`));
    if (types.size === 0 && token.length >= 3) {
      types = typeSet((code) =>
        cellTypeName(code)
          .toLowerCase()
          .split(/[^a-z0-9]+/)
          .some((word) => word.startsWith(token)),
      );
    }
    if (types.size === 0) return null;
    const set = types;
    preds.push((i) => set.has(neurons.cellType[i]));
  }
  return (i) => preds.every((p) => p(i));
}

export function search(
  data: SearchData,
  query: string,
  from: readonly [number, number, number],
  opts: { limit?: number; exclude?: number } = {},
): SearchResult {
  const match = compile(data, query);
  if (!match) return { hits: [], total: 0 };
  const { neurons, manifest } = data;
  const found: { i: number; d: number }[] = [];
  for (let i = 0; i < neurons.count; i++) {
    if (i === opts.exclude || !match(i)) continue;
    const d = Math.hypot(neurons.somaUm[i * 3] - from[0], neurons.somaUm[i * 3 + 1] - from[1], neurons.somaUm[i * 3 + 2] - from[2]);
    found.push({ i, d });
  }
  found.sort((a, b) => a.d - b.d || a.i - b.i);
  const hits = found.slice(0, opts.limit ?? MAX_RESULTS).map(({ i, d }): SearchHit => {
    const typeCode = manifest.cellTypes[neurons.cellType[i]] ?? "";
    return {
      neuron: i,
      typeCode,
      typeName: cellTypeName(typeCode),
      layer: neurons.layer[i],
      rootId: neurons.rootId[i].toString(),
      distanceUm: d,
    };
  });
  return { hits, total: found.length };
}

/**
 * The nearest match, for a chip. It skips `avoid` (the selected cell) so pressing a chip again moves
 * on to the next cell, and falls back to `avoid` when that is the only match.
 */
export function nearestMatch(data: SearchData, query: string, from: readonly [number, number, number], avoid: number): SearchHit | null {
  const first = search(data, query, from, { limit: 1, exclude: avoid }).hits[0];
  if (first || avoid < 0) return first ?? null;
  return search(data, query, from, { limit: 1 }).hits[0] ?? null;
}
