export interface ProjectionResult {
  coordinates: Array<[number, number, number]>;
  explained: [number, number, number];
}

const EPSILON = 1e-9;
export type VectorRow = ArrayLike<number>;

export async function projectPca(vectors: VectorRow[], dimensions = 3): Promise<ProjectionResult> {
  if (vectors.length === 0) {
    return { coordinates: [], explained: [0, 0, 0] };
  }

  if (vectors.length === 1) {
    return { coordinates: [[0, 0, 0]], explained: [1, 0, 0] };
  }

  const { rows: projected, explained } = await projectPcaMatrix(vectors, dimensions);
  const coordinates = projected.map((row) => toCoordinate(row));

  return {
    coordinates: normalizeCoordinates(coordinates),
    explained: [explained[0] ?? 0, explained[1] ?? 0, explained[2] ?? 0],
  };
}

export function normalizeCoordinates(coordinates: Array<[number, number, number]>) {
  let maxAbs = EPSILON;
  for (const [x, y, z] of coordinates) {
    maxAbs = Math.max(maxAbs, Math.abs(x), Math.abs(y), Math.abs(z));
  }
  const scale = 5 / maxAbs;
  return coordinates.map(([x, y, z]) => [x * scale, y * scale, z * scale] as [number, number, number]);
}

export function vectorsToMatrix(vectors: VectorRow[]) {
  const dimensions = vectors[0]?.length ?? 0;
  return vectors.map((row) => Array.from({ length: dimensions }, (_, index) => Number(row[index] ?? 0)));
}

export async function projectPcaMatrix(vectors: VectorRow[], dimensions: number) {
  const { PCA } = await import("ml-pca");
  const rows = vectorsToMatrix(vectors);
  const pca = new PCA(rows, { center: true, scale: false, method: "SVD" });
  const nComponents = Math.min(dimensions, rows.length, rows[0]?.length ?? dimensions);
  return {
    rows: pca.predict(rows, { nComponents }).to2DArray(),
    explained: pca.getExplainedVariance(),
  };
}

function toCoordinate(row: number[]) {
  return [row[0] ?? 0, row[1] ?? 0, row[2] ?? 0] as [number, number, number];
}
