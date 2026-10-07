import { describe, expect, it } from "vitest";
import { rotatePoint } from "./camera";

describe("3D camera", () => {
  it("rotates depth into the visible plane without changing vector length", () => {
    const point = rotatePoint(0, 0, 5, { yaw: Math.PI / 2, pitch: 0 });
    expect(point.x).toBeCloseTo(5);
    expect(point.y).toBeCloseTo(0);
    expect(point.depth).toBeCloseTo(0);
    const rotated = rotatePoint(1, 2, 3, { yaw: 0.7, pitch: -0.4 });
    expect(rotated.x ** 2 + rotated.y ** 2 + rotated.depth ** 2).toBeCloseTo(14);
  });
});
