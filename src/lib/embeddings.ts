import { AutoModelForCausalLM, AutoTokenizer, env, pipeline, type ProgressInfo } from "@huggingface/transformers";
import type { EmbeddingPoint, InputType, ModelPreset, OutputMode, PipelineStatus, TextSnippet } from "../types";
import { GROUP_COLORS, TOKEN_SAMPLE } from "../data";
import { projectPca } from "./pca";

env.allowRemoteModels = true;
env.allowLocalModels = false;
env.useBrowserCache = true;

type FeatureExtractor = Awaited<ReturnType<typeof pipeline<"feature-extraction">>>;
type CausalLmBundle = {
  tokenizer: Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>;
  model: Awaited<ReturnType<typeof AutoModelForCausalLM.from_pretrained>>;
};

const extractorCache = new Map<string, Promise<FeatureExtractor>>();
const causalLmCache = new Map<string, Promise<CausalLmBundle>>();

export async function createEmbeddingRun({
  model,
  outputMode,
  inputType,
  snippets,
  files,
  onStatus,
}: {
  model: ModelPreset;
  outputMode: OutputMode;
  inputType: InputType;
  snippets: TextSnippet[];
  files: File[];
  onStatus: (status: PipelineStatus) => void;
}) {
  validateCompatibility(model, inputType);

  const samples = await resolveSamples(inputType, snippets, files);
  if (samples.length < 2) {
    throw new Error("Add at least two inputs before running a projection.");
  }

  onStatus({ phase: "embedding", message: "Extracting embeddings", progress: 0.55 });
  const vectors = await extractVectors(model, samples.map((sample) => sample.text), outputMode, onStatus);

  onStatus({ phase: "projecting", message: "Projecting with PCA", progress: 0.82 });
  const projection = projectPca(vectors, 3);

  const points: EmbeddingPoint[] = samples.map((sample, index) => ({
    id: `${Date.now()}-${index}`,
    label: sample.label,
    snippet: sample.text,
    group: sample.group,
    source: sample.source,
    output: outputLabel(outputMode, model.task),
    vector: vectors[index],
    x: projection.coordinates[index][0],
    y: projection.coordinates[index][1],
    z: projection.coordinates[index][2],
  }));

  onStatus({
    phase: "ready",
    message: `Model loaded · ${points.length} embeddings · PCA projected`,
    progress: 1,
  });

  return {
    points,
    explained: projection.explained,
  };
}

async function getExtractor(modelId: string, onStatus: (status: PipelineStatus) => void) {
  if (!extractorCache.has(modelId)) {
    extractorCache.set(
      modelId,
      pipeline("feature-extraction", modelId, {
        dtype: "q8",
        progress_callback: (progress: ProgressInfo) => {
          if ("progress" in progress && typeof progress.progress === "number") {
            const file = "file" in progress && typeof progress.file === "string" ? progress.file : "";
            onStatus({
              phase: "loading",
              message: file ? `Loading ${file}` : "Loading model files",
              progress: Math.min(0.5, Math.max(0.08, progress.progress / 200)),
            });
          }
        },
      }) as Promise<FeatureExtractor>,
    );
  }

  return extractorCache.get(modelId)!;
}

async function getCausalLm(modelId: string, onStatus: (status: PipelineStatus) => void) {
  if (!causalLmCache.has(modelId)) {
    causalLmCache.set(
      modelId,
      Promise.all([
        AutoTokenizer.from_pretrained(modelId, {
          progress_callback: (progress: ProgressInfo) => reportProgress(progress, onStatus, 0.28),
        }),
        AutoModelForCausalLM.from_pretrained(modelId, {
          dtype: "q4",
          progress_callback: (progress: ProgressInfo) => reportProgress(progress, onStatus, 0.52),
        }),
      ]).then(([tokenizer, model]) => ({ tokenizer, model })),
    );
  }

  return causalLmCache.get(modelId)!;
}

async function extractVectors(
  model: ModelPreset,
  texts: string[],
  outputMode: OutputMode,
  onStatus: (status: PipelineStatus) => void,
) {
  onStatus({ phase: "loading", message: "Loading ONNX model", progress: 0.05 });

  if (model.task === "text-generation") {
    const lm = await getCausalLm(model.id, onStatus);
    onStatus({ phase: "embedding", message: "Pooling language-model layer states", progress: 0.62 });
    return extractCausalLmVectors(lm, texts, outputMode);
  }

  const extractor = await getExtractor(model.id, onStatus);

  if (outputMode === "tokens") {
    const output = await extractor(texts, { pooling: "mean", normalize: true });
    return tensorRows(output.tolist());
  }

  const pooling = outputMode === "hidden-4" ? "cls" : "mean";
  const output = await extractor(texts, { pooling, normalize: true });
  return tensorRows(output.tolist());
}

async function extractCausalLmVectors(bundle: CausalLmBundle, texts: string[], outputMode: OutputMode) {
  const inputs = bundle.tokenizer(texts, {
    padding: true,
    truncation: true,
    max_length: 96,
  });
  const outputs = await bundle.model(inputs);
  const layer = outputMode === "final" ? inferLastLayer(outputs) : 4;
  const valueState = outputs[`present.${layer}.value`];

  if (!valueState?.dims || valueState.dims.length !== 4) {
    throw new Error(`SmolLM2 did not return value states for layer ${layer}.`);
  }

  return normalizeRows(poolValueStates(valueState, inputs.attention_mask));
}

async function resolveSamples(inputType: InputType, snippets: TextSnippet[], files: File[]) {
  if (inputType === "tokens") {
    return TOKEN_SAMPLE.map((token) => ({
      text: token,
      label: token,
      group: "Tokens",
      source: "Token sample",
    }));
  }

  if (inputType === "files") {
    const loaded = await Promise.all(
      files.map(async (file) => ({
        text: trimText(await file.text(), 260),
        label: file.name,
        group: "Files",
        source: file.name,
      })),
    );
    return loaded.filter((file) => file.text.length > 0);
  }

  return snippets
    .filter((snippet) => snippet.text.trim().length > 0)
    .map((snippet) => ({
      text: snippet.text.trim(),
      label: snippet.label.trim() || trimText(snippet.text, 42),
      group: snippet.group,
      source: "Text snippets",
    }));
}

function tensorRows(value: unknown): number[][] {
  if (!Array.isArray(value)) {
    return [];
  }

  if (typeof value[0] === "number") {
    return [value as number[]];
  }

  return value.map((row) => {
    if (Array.isArray(row?.[0])) {
      return meanPool(row as number[][]);
    }
    return row as number[];
  });
}

function meanPool(rows: number[][]) {
  const dim = rows[0]?.length ?? 0;
  const pooled = Array.from({ length: dim }, () => 0);
  for (const row of rows) {
    for (let i = 0; i < dim; i += 1) {
      pooled[i] += row[i] ?? 0;
    }
  }
  return pooled.map((value) => value / Math.max(rows.length, 1));
}

function validateCompatibility(model: ModelPreset, inputType: InputType) {
  if (inputType === "images" && !model.supportsImages) {
    throw new Error("The selected model does not expose image embeddings. Choose an image-capable model first.");
  }

  if (model.task === "image-feature-extraction") {
    throw new Error(`${model.label} is image-capable, but image embedding extraction is not wired into this MVP path yet.`);
  }
}

function outputLabel(outputMode: OutputMode, task?: ModelPreset["task"]) {
  if (task === "text-generation") {
    if (outputMode === "final") return "Final LM value state";
    if (outputMode === "tokens") return "Token sample · LM layer 4";
    return "Layer 4 · LM value state";
  }
  if (outputMode === "hidden-4") return "Layer 4 · hidden state";
  if (outputMode === "tokens") return "Token table · embeddings";
  return "Final embedding";
}

function poolValueStates(valueState: { data: ArrayLike<number>; dims: number[] }, attentionMask?: { data: ArrayLike<number | bigint>; dims: number[] }) {
  const [batch, heads, sequence, headDim] = valueState.dims;
  const vectors: number[][] = [];

  for (let b = 0; b < batch; b += 1) {
    const vector = Array.from({ length: heads * headDim }, () => 0);
    let tokenCount = 0;

    for (let t = 0; t < sequence; t += 1) {
      const maskValue = attentionMask ? Number(attentionMask.data[b * sequence + t] ?? 0) : 1;
      if (maskValue === 0) continue;
      tokenCount += 1;

      for (let h = 0; h < heads; h += 1) {
        for (let d = 0; d < headDim; d += 1) {
          const sourceIndex = ((b * heads + h) * sequence + t) * headDim + d;
          vector[h * headDim + d] += Number(valueState.data[sourceIndex] ?? 0);
        }
      }
    }

    vectors.push(vector.map((value) => value / Math.max(tokenCount, 1)));
  }

  return vectors;
}

function normalizeRows(vectors: number[][]) {
  return vectors.map((vector) => {
    const magnitude = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0)) || 1;
    return vector.map((value) => value / magnitude);
  });
}

function inferLastLayer(outputs: Record<string, unknown>) {
  const layers = Object.keys(outputs)
    .map((key) => key.match(/^present\.(\d+)\.value$/)?.[1])
    .filter((value): value is string => Boolean(value))
    .map(Number);
  return layers.length ? Math.max(...layers) : 4;
}

function reportProgress(progress: ProgressInfo, onStatus: (status: PipelineStatus) => void, maxProgress: number) {
  if ("progress" in progress && typeof progress.progress === "number") {
    const file = "file" in progress && typeof progress.file === "string" ? progress.file : "";
    onStatus({
      phase: "loading",
      message: file ? `Loading ${file}` : "Loading model files",
      progress: Math.min(maxProgress, Math.max(0.08, progress.progress / 200)),
    });
  }
}

function trimText(value: string, maxLength: number) {
  const normalized = value.replace(/\s+/g, " ").trim();
  return normalized.length > maxLength ? `${normalized.slice(0, maxLength - 1)}…` : normalized;
}

export function runColor(index: number) {
  return ["#2563eb", "#f59e0b", "#f43f72", "#14b8a6", "#7c3aed"][index % 5];
}

export function colorForGroup(group: string) {
  return GROUP_COLORS[group] ?? "#64748b";
}
