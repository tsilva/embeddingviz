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
  parentLabel?: string;
  snippet: string;
  source: string;
  output: string;
  vector: ArrayLike<number>;
  x: number;
  y: number;
  z: number;
  kind?: "input" | "token" | "image";
  tokenId?: number;
  rawToken?: string;
  tokenCount?: number;
  chunkIndex?: number;
  chunkCount?: number;
  tokenStart?: number;
  tokenEnd?: number;
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
  task: "feature-extraction" | "text-generation" | "image-feature-extraction" | "clip-text";
  summary: string;
  recommendedOutput: OutputMode;
  outputModes: OutputMode[];
  maxInputTokens: number;
  supportsImages: boolean;
  note: string;
}

export interface PipelineStatus {
  phase: "idle" | "loading" | "embedding" | "projecting" | "ready" | "error";
  message: string;
  progress: number;
}

export interface PlannedEmbeddingSample {
  id: string;
  text: string;
  label: string;
  parentLabel: string;
  source: string;
  kind: "input";
  tokenCount: number;
  chunkIndex: number;
  chunkCount: number;
  tokenStart: number;
  tokenEnd: number;
}

export interface InputPlanItem {
  id: string;
  label: string;
  source: string;
  tokenCount: number;
  chunkCount: number;
  status: "ready" | "chunked" | "skipped";
  message: string;
}

export interface EmbeddingInputPlan {
  inputType: Exclude<InputType, "tokens">;
  modelId: string;
  chunkSize: number;
  overlapTokens: number;
  items: InputPlanItem[];
  samples: PlannedEmbeddingSample[];
  totalTokens: number;
  totalChunks: number;
  skippedCount: number;
}
