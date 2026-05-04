export interface ProjectionResult {
  coordinates: Array<[number, number, number]>;
  explained: [number, number, number];
}

const EPSILON = 1e-9;

export function projectPca(vectors: number[][], dimensions = 3): ProjectionResult {
  if (vectors.length === 0) {
    return { coordinates: [], explained: [0, 0, 0] };
  }

  if (vectors.length === 1) {
    return { coordinates: [[0, 0, 0]], explained: [1, 0, 0] };
  }

  const dim = vectors[0].length;
  const means = Array.from({ length: dim }, (_, col) => average(vectors.map((row) => row[col] ?? 0)));
  const centered = vectors.map((row) => row.map((value, col) => (value ?? 0) - means[col]));
  const covarianceRows = sampleRows(centered, 1600);
  const covariance = buildCovariance(covarianceRows, dim);
  const components: number[][] = [];
  const eigenvalues: number[] = [];
  let working = covariance.map((row) => [...row]);

  for (let i = 0; i < dimensions; i += 1) {
    const component = powerIteration(working, 80, i);
    const eigenvalue = Math.max(dot(component, multiplyMatrixVector(working, component)), 0);
    components.push(component);
    eigenvalues.push(eigenvalue);
    working = deflate(working, component, eigenvalue);
  }

  const coordinates = centered.map((row) => {
    const projected = components.map((component) => dot(row, component));
    return [projected[0] ?? 0, projected[1] ?? 0, projected[2] ?? 0] as [number, number, number];
  });

  const totalVariance = covariance.reduce((sum, row, index) => sum + row[index], 0) || EPSILON;
  const explained = eigenvalues.map((value) => value / totalVariance);
  return {
    coordinates: normalizeCoordinates(coordinates),
    explained: [explained[0] ?? 0, explained[1] ?? 0, explained[2] ?? 0],
  };
}

function sampleRows(rows: number[][], maxRows: number) {
  if (rows.length <= maxRows) {
    return rows;
  }

  const step = rows.length / maxRows;
  return Array.from({ length: maxRows }, (_, index) => rows[Math.floor(index * step)]);
}

function buildCovariance(centered: number[][], dim: number) {
  const matrix = Array.from({ length: dim }, () => Array.from({ length: dim }, () => 0));
  const scale = 1 / Math.max(centered.length - 1, 1);

  for (const row of centered) {
    for (let i = 0; i < dim; i += 1) {
      const left = row[i] ?? 0;
      for (let j = i; j < dim; j += 1) {
        matrix[i][j] += left * (row[j] ?? 0) * scale;
      }
    }
  }

  for (let i = 0; i < dim; i += 1) {
    for (let j = 0; j < i; j += 1) {
      matrix[i][j] = matrix[j][i];
    }
  }

  return matrix;
}

function powerIteration(matrix: number[][], iterations: number, seed: number) {
  let vector = Array.from({ length: matrix.length }, (_, index) => {
    const value = Math.sin((index + 1) * (seed + 1) * 12.9898) * 43758.5453;
    return value - Math.floor(value) - 0.5;
  });
  vector = normalize(vector);

  for (let i = 0; i < iterations; i += 1) {
    vector = normalize(multiplyMatrixVector(matrix, vector));
  }

  return vector;
}

function multiplyMatrixVector(matrix: number[][], vector: number[]) {
  return matrix.map((row) => dot(row, vector));
}

function deflate(matrix: number[][], vector: number[], eigenvalue: number) {
  return matrix.map((row, i) => row.map((value, j) => value - eigenvalue * vector[i] * vector[j]));
}

function normalize(vector: number[]) {
  const magnitude = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0)) || EPSILON;
  return vector.map((value) => value / magnitude);
}

function normalizeCoordinates(coordinates: Array<[number, number, number]>) {
  const maxAbs = Math.max(
    ...coordinates.flatMap(([x, y, z]) => [Math.abs(x), Math.abs(y), Math.abs(z)]),
    EPSILON,
  );
  const scale = 5 / maxAbs;
  return coordinates.map(([x, y, z]) => [x * scale, y * scale, z * scale] as [number, number, number]);
}

function average(values: number[]) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);
}

function dot(a: number[], b: number[]) {
  return a.reduce((sum, value, index) => sum + value * (b[index] ?? 0), 0);
}
