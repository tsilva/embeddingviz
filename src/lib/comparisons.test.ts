import { describe, expect, it } from "vitest";
import { cosineSimilarity, embeddingSpaceId, nearestNeighborsForPoint } from "./comparisons";
import { MODEL_PRESETS } from "../data";
import type { EmbeddingPoint, RunRecord } from "../types";

const point = (id: string, vector: number[]): EmbeddingPoint => ({ id, vector, label: id, snippet: id, source: "test", output: "final", x: 0, y: 0, z: 0 });
const run = (id: string, spaceId: string, points: EmbeddingPoint[], visible = true): RunRecord => ({ id, spaceId, points, visible, name: id, model: id, output: "final", reduction: "PCA", color: "#2563eb", count: points.length });

describe("embedding comparisons", () => {
  it("rejects mismatched dimensions instead of truncating", () => {
    expect(cosineSimilarity([1, 0], [1, 0, 3])).toBe(-Infinity);
    expect(cosineSimilarity([0, 0], [1, 0])).toBe(-Infinity);
    expect(cosineSimilarity([1, 0], [1, 0])).toBe(1);
  });

  it("includes compatible saved runs but excludes different spaces even with equal dimensions", () => {
    const selected = point("anchor", [1, 0]);
    const runs = [run("current", "model-a/final", [selected, point("mid", [0.8, 0.6])]),
      run("previous", "model-a/final", [point("close", [0.99, 0.1])], false),
      run("other-model", "model-b/final", [point("invalid-model", [1, 0])]),
      run("other-layer", "model-a/layer-1", [point("invalid-layer", [1, 0])]),
      run("bad-shape", "model-a/final", [point("invalid-shape", [1, 0, 3])])];
    expect(nearestNeighborsForPoint(selected, runs, 10).map(({ point }) => point.id)).toEqual(["close", "mid"]);
  });

  it("distinguishes model, precision and output identities", () => {
    const model = MODEL_PRESETS[0];
    expect(embeddingSpaceId(model, "final")).not.toBe(embeddingSpaceId(MODEL_PRESETS[1], "final"));
    expect(embeddingSpaceId(model, "final")).not.toBe(embeddingSpaceId(model, "hidden-1"));
    expect(embeddingSpaceId(model, "final")).not.toBe(embeddingSpaceId({ ...model, dtype: "fp32" }, "final"));
  });
});
