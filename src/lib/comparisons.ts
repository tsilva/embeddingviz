import type { EmbeddingPoint, ModelPreset, OutputMode, RunRecord } from "../types";

// Model/output and preprocessing must match; dimensions alone do not define a space.
export function embeddingSpaceId(model: ModelPreset, output: OutputMode) {
  return JSON.stringify([model.id, model.dtype ?? "q8", output, "token-chunks-v1", model.task]);
}

export function cosineSimilarity(a: ArrayLike<number>, b: ArrayLike<number>) {
  if (!a.length || a.length !== b.length) return Number.NEGATIVE_INFINITY;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let index = 0; index < a.length; index += 1) {
    dot += a[index] * b[index];
    normA += a[index] * a[index];
    normB += b[index] * b[index];
  }
  if (!normA || !normB) return Number.NEGATIVE_INFINITY;
  return Math.min(1, Math.max(-1, dot / Math.sqrt(normA * normB)));
}

export function nearestNeighborsForPoint(selected: EmbeddingPoint, runs: RunRecord[], limit: number) {
  const origin = runs.find((run) => run.points.some((point) => point.id === selected.id));
  if (!origin) return [];
  return runs.filter((run) => run.spaceId === origin.spaceId)
    .flatMap((run) => run.points.filter((point) => point.id !== selected.id).map((point) => {
      const similarity = cosineSimilarity(selected.vector, point.vector);
      return { run, point, similarity, distance: 1 - similarity };
    }))
    .filter(({ similarity }) => Number.isFinite(similarity))
    .sort((a, b) => b.similarity - a.similarity || a.point.label.localeCompare(b.point.label))
    .slice(0, limit);
}
