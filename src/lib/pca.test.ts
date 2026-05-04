import { describe, expect, it } from "vitest";
import { normalizeCoordinates, projectPca } from "./pca";

describe("PCA projection", () => {
  it("normalizes projected coordinates to a stable plot radius", () => {
    const normalized = normalizeCoordinates([
      [2, -4, 1],
      [-1, 0, 0.5],
    ]);

    expect(normalized).toEqual([
      [2.5, -5, 1.25],
      [-1.25, 0, 0.625],
    ]);
  });

  it("keeps degenerate coordinates finite during normalization", () => {
    expect(normalizeCoordinates([[0, 0, 0]])).toEqual([[0, 0, 0]]);
  });

  it("returns normalized finite coordinates from PCA", () => {
    const result = projectPca([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ]);

    expect(result.coordinates).toHaveLength(3);
    for (const coordinate of result.coordinates) {
      expect(coordinate.every(Number.isFinite)).toBe(true);
      expect(Math.max(...coordinate.map(Math.abs))).toBeLessThanOrEqual(5);
    }
  });
});
