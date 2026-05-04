import type { PipelineStatus, ReductionMethod } from "../types";
import { normalizeCoordinates, projectPca, type ProjectionResult, type VectorRow } from "./pca";

type StatusReporter = (status: PipelineStatus) => void;

interface Neighbor {
  index: number;
  distance: number;
}

const EPSILON = 1e-6;

export async function projectReductionCore(vectors: VectorRow[], method: ReductionMethod, onStatus: StatusReporter): Promise<ProjectionResult> {
  onStatus({ phase: "projecting", message: `Projecting with ${method}`, progress: 0.82 });
  await yieldToBrowser();

  const seed = projectPca(vectors, 3);
  if (method === "PCA" || vectors.length < 3) {
    return seed;
  }

  onStatus({ phase: "projecting", message: `Building ${method} neighborhood graph`, progress: 0.86 });
  await yieldToBrowser();

  const neighbors = buildApproximateNeighbors(seed.coordinates, method === "UMAP" ? 10 : 14);
  const layout = method === "UMAP" ? await runUmapLayout(seed.coordinates, neighbors, onStatus) : await runTsneLayout(seed.coordinates, neighbors, onStatus);

  return {
    coordinates: normalizeCoordinates(layout),
    explained: [0, 0, 0],
  };
}

async function runUmapLayout(
  coordinates: Array<[number, number, number]>,
  neighbors: Neighbor[][],
  onStatus: StatusReporter,
): Promise<Array<[number, number, number]>> {
  const state = createLayoutState(coordinates, 0.025);
  const iterations = iterationCount(coordinates.length, 56, 26, 16);

  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const decay = 1 - iteration / Math.max(iterations, 1);
    const step = 0.028 * decay + 0.004;
    for (let i = 0; i < coordinates.length; i += 1) {
      const x = state.x[i];
      const y = state.y[i];
      let fx = 0;
      let fy = 0;

      for (const neighbor of neighbors[i]) {
        const j = neighbor.index;
        const dx = state.x[j] - x;
        const dy = state.y[j] - y;
        const distance = Math.sqrt(dx * dx + dy * dy + EPSILON);
        const target = Math.max(0.08, Math.min(1.8, neighbor.distance * 0.72));
        const force = (distance - target) / distance;
        fx += dx * force * 0.34;
        fy += dy * force * 0.34;
      }

      for (let sample = 0; sample < 2; sample += 1) {
        const j = hashedIndex(i, iteration, sample, coordinates.length);
        if (j === i) continue;
        const dx = x - state.x[j];
        const dy = y - state.y[j];
        const distanceSq = dx * dx + dy * dy + 0.08;
        const force = Math.min(0.35, 0.018 / distanceSq);
        fx += dx * force;
        fy += dy * force;
      }

      state.x[i] += clamp(fx, -1, 1) * step;
      state.y[i] += clamp(fy, -1, 1) * step;
    }

    if (iteration % 4 === 3) {
      onStatus({ phase: "projecting", message: "Optimizing UMAP layout", progress: 0.88 + (iteration / iterations) * 0.09 });
      await yieldToBrowser();
    }
  }

  return materializeLayout(state, coordinates, 0.72, "UMAP");
}

async function runTsneLayout(
  coordinates: Array<[number, number, number]>,
  neighbors: Neighbor[][],
  onStatus: StatusReporter,
): Promise<Array<[number, number, number]>> {
  const state = createLayoutState(coordinates, 0.045);
  const iterations = iterationCount(coordinates.length, 72, 32, 20);

  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const exaggeration = iteration < iterations * 0.38 ? 1.8 : 1;
    const step = 0.024 * (1 - iteration / Math.max(iterations, 1)) + 0.0035;
    for (let i = 0; i < coordinates.length; i += 1) {
      const x = state.x[i];
      const y = state.y[i];
      let fx = 0;
      let fy = 0;

      for (const neighbor of neighbors[i]) {
        const j = neighbor.index;
        const dx = state.x[j] - x;
        const dy = state.y[j] - y;
        const distanceSq = dx * dx + dy * dy + 0.04;
        const similarity = Math.exp(-neighbor.distance * neighbor.distance * 0.42);
        const force = (similarity * exaggeration) / (1 + distanceSq);
        fx += dx * force * 0.12;
        fy += dy * force * 0.12;
      }

      for (let sample = 0; sample < 3; sample += 1) {
        const j = hashedIndex(i, iteration, sample + 7, coordinates.length);
        if (j === i) continue;
        const dx = x - state.x[j];
        const dy = y - state.y[j];
        const distanceSq = dx * dx + dy * dy + 0.12;
        const force = Math.min(0.42, 0.032 / distanceSq);
        fx += dx * force;
        fy += dy * force;
      }

      state.x[i] += clamp(fx, -1.2, 1.2) * step;
      state.y[i] += clamp(fy, -1.2, 1.2) * step;
    }

    if (iteration % 4 === 3) {
      onStatus({ phase: "projecting", message: "Optimizing t-SNE layout", progress: 0.88 + (iteration / iterations) * 0.09 });
      await yieldToBrowser();
    }
  }

  return materializeLayout(state, coordinates, 0.5, "t-SNE");
}

function buildApproximateNeighbors(coordinates: Array<[number, number, number]>, neighborCount: number): Neighbor[][] {
  const bins = new Map<string, number[]>();
  const binCount = coordinates.length > 10000 ? 28 : 18;

  coordinates.forEach(([x, y], index) => {
    const key = binKey(x, y, binCount);
    const values = bins.get(key);
    if (values) {
      values.push(index);
    } else {
      bins.set(key, [index]);
    }
  });

  return coordinates.map((coordinate, index) => {
    const candidates = nearbyCandidates(coordinate, binCount, bins, index);
    for (let sample = 0; sample < neighborCount * 2 && candidates.length < neighborCount * 3; sample += 1) {
      candidates.push(hashedIndex(index, sample, 17, coordinates.length));
    }

    return candidates
      .filter((candidate) => candidate !== index)
      .map((candidate) => ({
        index: candidate,
        distance: distance3(coordinate, coordinates[candidate]),
      }))
      .sort((left, right) => left.distance - right.distance)
      .slice(0, neighborCount);
  });
}

function nearbyCandidates(
  [x, y]: [number, number, number],
  binCount: number,
  bins: Map<string, number[]>,
  index: number,
) {
  const { bx, by } = binPosition(x, y, binCount);
  const candidates: number[] = [];
  for (let ox = -1; ox <= 1; ox += 1) {
    for (let oy = -1; oy <= 1; oy += 1) {
      const values = bins.get(`${bx + ox}:${by + oy}`) ?? [];
      const stride = Math.max(1, Math.floor(values.length / 96));
      const start = values.length ? hashedIndex(index, ox + 3, oy + 5, values.length) % stride : 0;
      for (let i = start; i < values.length; i += stride) {
        candidates.push(values[i]);
      }
    }
  }
  return candidates;
}

function createLayoutState(coordinates: Array<[number, number, number]>, noise: number) {
  const x = new Float32Array(coordinates.length);
  const y = new Float32Array(coordinates.length);
  coordinates.forEach(([cx, cy], index) => {
    const jitter = hashUnit(index + 1) - 0.5;
    x[index] = cx * 0.78 + jitter * noise;
    y[index] = cy * 0.78 + (hashUnit(index + 101) - 0.5) * noise;
  });
  return { x, y };
}

function materializeLayout(
  state: { x: Float32Array; y: Float32Array },
  coordinates: Array<[number, number, number]>,
  zScale: number,
  method: Exclude<ReductionMethod, "PCA">,
): Array<[number, number, number]> {
  return coordinates.map(([baseX, baseY, z], index) => {
    const x = state.x[index];
    const y = state.y[index];
    const radius = Math.sqrt(x * x + y * y + EPSILON);
    const angle = Math.atan2(y, x);

    if (method === "UMAP") {
      const shapedRadius = Math.pow(radius, 0.86) * 1.12;
      const shapedAngle = angle + Math.sin(z * 1.7 + baseX * 0.25) * 0.12;
      return [Math.cos(shapedAngle) * shapedRadius, Math.sin(shapedAngle) * shapedRadius, z * zScale] as [number, number, number];
    }

    const islandRadius = Math.log1p(radius * 1.4) * 2.25;
    const islandAngle = angle + Math.sin(z * 2.3 + radius) * 0.32;
    return [
      Math.cos(islandAngle) * islandRadius + Math.tanh(baseX) * 0.45,
      Math.sin(islandAngle) * islandRadius + Math.tanh(baseY) * 0.45,
      z * zScale,
    ] as [number, number, number];
  });
}

function binPosition(x: number, y: number, binCount: number) {
  const bx = Math.max(0, Math.min(binCount - 1, Math.floor(((x + 5) / 10) * binCount)));
  const by = Math.max(0, Math.min(binCount - 1, Math.floor(((y + 5) / 10) * binCount)));
  return { bx, by };
}

function binKey(x: number, y: number, binCount: number) {
  const { bx, by } = binPosition(x, y, binCount);
  return `${bx}:${by}`;
}

function distance3(left: [number, number, number], right: [number, number, number]) {
  const dx = left[0] - right[0];
  const dy = left[1] - right[1];
  const dz = left[2] - right[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function iterationCount(length: number, small: number, medium: number, large: number) {
  if (length > 20000) return large;
  if (length > 5000) return medium;
  return small;
}

function hashedIndex(index: number, iteration: number, sample: number, length: number) {
  if (length <= 1) return 0;
  const value = Math.sin((index + 1) * 12.9898 + (iteration + 1) * 78.233 + (sample + 1) * 37.719) * 43758.5453;
  return Math.abs(Math.floor(value)) % length;
}

function hashUnit(value: number) {
  const hashed = Math.sin(value * 12.9898) * 43758.5453;
  return hashed - Math.floor(hashed);
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function yieldToBrowser() {
  return new Promise<void>((resolve) => {
    if (typeof window !== "undefined" && "requestAnimationFrame" in window) {
      window.requestAnimationFrame(() => resolve());
      return;
    }
    setTimeout(resolve, 0);
  });
}
