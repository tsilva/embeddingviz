import type { ReductionMethod } from "../types";

export const MAX_TSNE_POINTS = 1000;

export function validateReductionSize(count: number, method: ReductionMethod) {
  if (method === "t-SNE" && count > MAX_TSNE_POINTS) {
    throw new Error(`t-SNE supports at most ${MAX_TSNE_POINTS.toLocaleString()} points; this run has ${count.toLocaleString()}. Choose PCA or UMAP, or reduce the inputs.`);
  }
}
