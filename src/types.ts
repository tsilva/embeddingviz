export type InputType = "text" | "files" | "tokens";
export type ReductionMethod = "PCA" | "UMAP" | "t-SNE";
export type OutputMode = "final" | "hidden-4" | "tokens";

export interface TextSnippet {
  id: string;
  text: string;
}

export interface EmbeddingPoint {
  id: string;
  label: string;
  snippet: string;
  source: string;
  output: string;
  vector: ArrayLike<number>;
  x: number;
  y: number;
  z: number;
  kind?: "input" | "token";
  tokenId?: number;
  rawToken?: string;
}

export interface RunRecord {
  id: string;
  name: string;
  model: string;
  output: string;
  reduction: ReductionMethod;
  color: string;
  count: number;
  current?: boolean;
  visible: boolean;
  points: EmbeddingPoint[];
}

export interface ModelPreset {
  id: string;
  label: string;
  task: "feature-extraction" | "text-generation" | "image-feature-extraction";
  summary: string;
  recommendedOutput: OutputMode;
  outputModes: OutputMode[];
  supportsImages: boolean;
  note: string;
}

export interface PipelineStatus {
  phase: "idle" | "loading" | "embedding" | "projecting" | "ready" | "error";
  message: string;
  progress: number;
}
