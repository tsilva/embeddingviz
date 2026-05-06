import type { PipelineStatus, ReductionMethod } from "../types";
import { normalizeCoordinates, projectPca, projectPcaMatrix, vectorsToMatrix, type ProjectionResult, type VectorRow } from "./pca";

type StatusReporter = (status: PipelineStatus) => void;

const TSNE_TARGET_DIMENSIONS = 50;
const TSNE_EARLY_EXAGGERATION_ITERATIONS = 250;
const TSNE_EARLY_EXAGGERATION = 12;
const TSNE_EPSILON = 1e-12;

export async function projectReductionCore(vectors: VectorRow[], method: ReductionMethod, onStatus: StatusReporter): Promise<ProjectionResult> {
  onStatus({ phase: "projecting", message: `Projecting with ${method}`, progress: 0.82 });
  await yieldToBrowser();

  if (method === "PCA" || vectors.length < 4) {
    return await projectPca(vectors, 3);
  }

  const matrix = vectorsToMatrix(vectors);
  if (method === "UMAP") {
    return projectUmap(matrix, onStatus);
  }

  return projectTsne(matrix, onStatus);
}

async function projectUmap(vectors: number[][], onStatus: StatusReporter): Promise<ProjectionResult> {
  onStatus({ phase: "projecting", message: "Building UMAP neighborhood graph", progress: 0.84 });
  await yieldToBrowser();

  const { UMAP } = await import("umap-js");
  const umap = new UMAP({
    distanceFn: cosineDistance,
    minDist: 0.1,
    nComponents: 3,
    nEpochs: umapEpochs(vectors.length),
    nNeighbors: Math.max(2, Math.min(15, vectors.length - 1)),
    random: seededRandom(42),
  });

  const embedding = await umap.fitAsync(vectors, (epoch) => {
    if (epoch % 10 === 0) {
      onStatus({ phase: "projecting", message: "Optimizing UMAP layout", progress: Math.min(0.97, 0.86 + epoch / umapEpochs(vectors.length) * 0.11) });
    }
  });

  return {
    coordinates: normalizeCoordinates(embedding.map(toCoordinate)),
    explained: [0, 0, 0],
  };
}

async function projectTsne(vectors: number[][], onStatus: StatusReporter): Promise<ProjectionResult> {
  onStatus({ phase: "projecting", message: "Preparing t-SNE distances", progress: 0.84 });
  await yieldToBrowser();

  const prepared = await prepareTsneInput(vectors);
  const probabilities = jointProbabilities(prepared, tsnePerplexity(vectors.length));
  const embedding = initialTsneEmbedding(prepared.length);
  const updates = Array.from({ length: prepared.length }, () => [0, 0]);
  const gains = Array.from({ length: prepared.length }, () => [1, 1]);
  const iterations = tsneIterations(vectors.length);
  const learningRate = Math.max(200, vectors.length / 12);

  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const exaggeration = iteration < TSNE_EARLY_EXAGGERATION_ITERATIONS ? TSNE_EARLY_EXAGGERATION : 1;
    const momentum = iteration < TSNE_EARLY_EXAGGERATION_ITERATIONS ? 0.5 : 0.8;
    const gradients = tsneGradients(embedding, probabilities, exaggeration);

    for (let i = 0; i < embedding.length; i += 1) {
      for (let dim = 0; dim < 2; dim += 1) {
        gains[i][dim] = Math.sign(gradients[i][dim]) !== Math.sign(updates[i][dim]) ? gains[i][dim] + 0.2 : Math.max(gains[i][dim] * 0.8, 0.01);
        updates[i][dim] = momentum * updates[i][dim] - learningRate * gains[i][dim] * gradients[i][dim];
        embedding[i][dim] += updates[i][dim];
      }
    }

    zeroMean(embedding);
    if (iteration % 25 === 24) {
      onStatus({ phase: "projecting", message: "Optimizing t-SNE layout", progress: 0.86 + (iteration / Math.max(iterations, 1)) * 0.11 });
      await yieldToBrowser();
    }
  }

  return {
    coordinates: normalizeCoordinates(embedding.map(toCoordinate)),
    explained: [0, 0, 0],
  };
}

async function prepareTsneInput(vectors: number[][]) {
  const dimensions = vectors[0]?.length ?? 0;
  const componentLimit = Math.min(TSNE_TARGET_DIMENSIONS, vectors.length - 1, dimensions);
  if (componentLimit < 2 || dimensions <= TSNE_TARGET_DIMENSIONS) {
    return vectors;
  }

  return (await projectPcaMatrix(vectors, componentLimit)).rows;
}

function umapEpochs(length: number) {
  if (length > 10000) return 200;
  if (length > 1000) return 350;
  return 500;
}

function tsneIterations(length: number) {
  if (length > 5000) return 500;
  if (length > 1000) return 750;
  return 1000;
}

function tsnePerplexity(length: number) {
  return Math.max(2, Math.min(30, Math.floor((length - 1) / 3)));
}

function jointProbabilities(vectors: number[][], perplexity: number) {
  const distances = pairwiseSquaredDistances(vectors);
  const conditional = distances.map((row, index) => conditionalProbabilities(row, index, perplexity));
  const count = vectors.length;
  const probabilities = Array.from({ length: count }, () => Array.from({ length: count }, () => 0));

  for (let i = 0; i < count; i += 1) {
    for (let j = 0; j < count; j += 1) {
      if (i === j) continue;
      probabilities[i][j] = Math.max((conditional[i][j] + conditional[j][i]) / (2 * count), TSNE_EPSILON);
    }
  }

  return probabilities;
}

function conditionalProbabilities(distances: number[], selfIndex: number, perplexity: number) {
  const targetEntropy = Math.log(perplexity);
  let beta = 1;
  let betaMin = -Infinity;
  let betaMax = Infinity;
  let probabilities = probabilityDistribution(distances, selfIndex, beta);

  for (let attempt = 0; attempt < 50; attempt += 1) {
    const entropy = shannonEntropy(distances, selfIndex, beta);
    const entropyDiff = entropy - targetEntropy;
    if (Math.abs(entropyDiff) < 1e-5) break;

    if (entropyDiff > 0) {
      betaMin = beta;
      beta = Number.isFinite(betaMax) ? (beta + betaMax) / 2 : beta * 2;
    } else {
      betaMax = beta;
      beta = Number.isFinite(betaMin) ? (beta + betaMin) / 2 : beta / 2;
    }

    probabilities = probabilityDistribution(distances, selfIndex, beta);
  }

  return probabilities;
}

function probabilityDistribution(distances: number[], selfIndex: number, beta: number) {
  const probabilities = distances.map((distance, index) => (index === selfIndex ? 0 : Math.exp(-distance * beta)));
  const sum = probabilities.reduce((total, value) => total + value, 0) || TSNE_EPSILON;
  return probabilities.map((value) => value / sum);
}

function shannonEntropy(distances: number[], selfIndex: number, beta: number) {
  let weightedDistance = 0;
  let sum = 0;
  for (let index = 0; index < distances.length; index += 1) {
    if (index === selfIndex) continue;
    const unnormalized = Math.exp(-distances[index] * beta);
    weightedDistance += distances[index] * unnormalized;
    sum += unnormalized;
  }
  return Math.log(sum || TSNE_EPSILON) + (beta * weightedDistance) / (sum || TSNE_EPSILON);
}

function pairwiseSquaredDistances(vectors: number[][]) {
  return vectors.map((left, leftIndex) =>
    vectors.map((right, rightIndex) => {
      if (leftIndex === rightIndex) return 0;
      let distance = 0;
      for (let dim = 0; dim < left.length; dim += 1) {
        const delta = (left[dim] ?? 0) - (right[dim] ?? 0);
        distance += delta * delta;
      }
      return distance;
    }),
  );
}

function initialTsneEmbedding(length: number) {
  const random = seededRandom(987);
  return Array.from({ length }, () => [(random() - 0.5) * 1e-4, (random() - 0.5) * 1e-4]);
}

function tsneGradients(embedding: number[][], probabilities: number[][], exaggeration: number) {
  const count = embedding.length;
  const numerators = Array.from({ length: count }, () => Array.from({ length: count }, () => 0));
  let qSum = 0;

  for (let i = 0; i < count; i += 1) {
    for (let j = i + 1; j < count; j += 1) {
      const dx = embedding[i][0] - embedding[j][0];
      const dy = embedding[i][1] - embedding[j][1];
      const numerator = 1 / (1 + dx * dx + dy * dy);
      numerators[i][j] = numerator;
      numerators[j][i] = numerator;
      qSum += numerator * 2;
    }
  }

  const gradients = Array.from({ length: count }, () => [0, 0]);
  for (let i = 0; i < count; i += 1) {
    for (let j = 0; j < count; j += 1) {
      if (i === j) continue;
      const q = Math.max(numerators[i][j] / (qSum || TSNE_EPSILON), TSNE_EPSILON);
      const multiplier = 4 * (probabilities[i][j] * exaggeration - q) * numerators[i][j];
      gradients[i][0] += multiplier * (embedding[i][0] - embedding[j][0]);
      gradients[i][1] += multiplier * (embedding[i][1] - embedding[j][1]);
    }
  }
  return gradients;
}

function zeroMean(embedding: number[][]) {
  const mean = embedding.reduce(
    (sum, point) => {
      sum[0] += point[0];
      sum[1] += point[1];
      return sum;
    },
    [0, 0],
  );
  mean[0] /= Math.max(embedding.length, 1);
  mean[1] /= Math.max(embedding.length, 1);
  for (const point of embedding) {
    point[0] -= mean[0];
    point[1] -= mean[1];
  }
}

function toCoordinate(row: number[]) {
  return [row[0] ?? 0, row[1] ?? 0, row[2] ?? 0] as [number, number, number];
}

function cosineDistance(left: number[], right: number[]) {
  let dot = 0;
  let leftMagnitude = 0;
  let rightMagnitude = 0;
  for (let index = 0; index < left.length; index += 1) {
    const leftValue = left[index] ?? 0;
    const rightValue = right[index] ?? 0;
    dot += leftValue * rightValue;
    leftMagnitude += leftValue * leftValue;
    rightMagnitude += rightValue * rightValue;
  }

  if (leftMagnitude === 0 || rightMagnitude === 0) return 1;
  return 1 - dot / Math.sqrt(leftMagnitude * rightMagnitude);
}

function seededRandom(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0x100000000;
  };
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
