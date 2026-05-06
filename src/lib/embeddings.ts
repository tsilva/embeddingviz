import type { ProgressInfo } from "@huggingface/transformers";
import type {
  EmbeddingInputPlan,
  EmbeddingPoint,
  InputPlanItem,
  InputType,
  ModelPreset,
  OutputMode,
  PipelineStatus,
  PlannedEmbeddingSample,
  ReductionMethod,
  TextSnippet,
} from "../types";

type TransformersModule = typeof import("@huggingface/transformers");
type FeatureExtractor = (texts: string[], options: { pooling: string; normalize: boolean }) => Promise<{ tolist(): unknown }>;
type ImageFeatureExtractor = (images: File[], options?: { pool?: boolean }) => Promise<{ tolist(): unknown }>;
type Tokenizer = {
  encode: (text: string, options?: { add_special_tokens?: boolean }) => number[];
  decode: (ids: number[], options?: { skip_special_tokens?: boolean }) => string;
  get_vocab: () => Map<string, number>;
  (texts: string[], options: { padding: boolean; truncation: boolean; max_length: number }): {
    attention_mask?: { data: ArrayLike<number | bigint>; dims: number[] };
    [key: string]: unknown;
  };
};
type CausalLmModel = (inputs: ReturnType<Tokenizer>) => Promise<Record<string, { data: ArrayLike<number>; dims: number[] } | unknown>>;
type CausalLmBundle = {
  tokenizer: Tokenizer;
  model: CausalLmModel;
};
type ClipTextModel = (inputs: ReturnType<Tokenizer>) => Promise<Record<string, { data: ArrayLike<number>; dims: number[] } | unknown>>;
type ClipTextBundle = {
  tokenizer: Tokenizer;
  model: ClipTextModel;
};
type ImageProcessor = (images: unknown[] | unknown) => Promise<Record<string, unknown>>;
type ClipVisionModel = (inputs: Record<string, unknown>) => Promise<Record<string, { data: ArrayLike<number>; dims: number[] } | unknown>>;
type ClipVisionBundle = {
  processor: ImageProcessor;
  model: ClipVisionModel;
  RawImage: { fromBlob(blob: Blob): Promise<unknown> };
};
type EmbeddingWorkerMessage =
  | { id: string; type: "status"; status: PipelineStatus }
  | { id: string; type: "result"; rows: number; dimensions: number; buffer: ArrayBuffer }
  | { id: string; type: "error"; message: string };
const MAX_TOKEN_POINTS = 50000;
const CHUNK_OVERLAP_TOKENS = 24;

export interface ResolvedSample {
  text: string;
  label: string;
  parentLabel?: string;
  source: string;
  kind?: "input" | "token" | "image";
  image?: File;
  tokenId?: number;
  rawToken?: string;
  tokenCount?: number;
  chunkIndex?: number;
  chunkCount?: number;
  tokenStart?: number;
  tokenEnd?: number;
}

const extractorCache = new Map<string, Promise<FeatureExtractor>>();
const imageExtractorCache = new Map<string, Promise<ImageFeatureExtractor>>();
const causalLmCache = new Map<string, Promise<CausalLmBundle>>();
const clipTextCache = new Map<string, Promise<ClipTextBundle>>();
const clipVisionCache = new Map<string, Promise<ClipVisionBundle>>();
const tokenizerCache = new Map<string, Promise<Tokenizer>>();
let transformersPromise: Promise<TransformersModule> | null = null;

async function loadTransformers() {
  if (!transformersPromise) {
    transformersPromise = import("@huggingface/transformers").then((transformers) => {
      transformers.env.allowRemoteModels = true;
      transformers.env.allowLocalModels = false;
      transformers.env.useBrowserCache = true;
      return transformers;
    });
  }

  return transformersPromise;
}

export async function createEmbeddingRun({
  model,
  outputMode,
  inputType,
  reduction,
  snippets,
  files,
  inputPlan,
  onStatus,
}: {
  model: ModelPreset;
  outputMode: OutputMode;
  inputType: InputType;
  reduction: ReductionMethod;
  snippets: TextSnippet[];
  files: File[];
  inputPlan?: EmbeddingInputPlan | null;
  onStatus: (status: PipelineStatus) => void;
}) {
  validateCompatibility(model, inputType, outputMode);
  validateFiles(model, inputType, files);

  const samples =
    inputType === "tokens" || model.task === "image-feature-extraction" || model.task === "clip-text"
      ? await resolveSamples(inputType, snippets, files, model, onStatus)
      : resolvePlannedSamples(inputPlan);
  if (samples.length < 2) {
    throw new Error("Add at least two inputs before running a projection.");
  }

  onStatus({ phase: "embedding", message: "Extracting embeddings", progress: 0.55 });
  const vectors = await extractVectors(model, samples, inputType, outputMode, onStatus);

  const { projectReduction } = await import("./reductions");
  const projection = await projectReduction(vectors, reduction, onStatus);

  const points: EmbeddingPoint[] = samples.map((sample, index) => ({
    id: `${Date.now()}-${index}`,
    label: sample.label,
    parentLabel: sample.parentLabel,
    snippet: sample.text,
    source: sample.source,
    output: outputLabel(outputMode, model.task),
    vector: vectors[index],
    x: projection.coordinates[index][0],
    y: projection.coordinates[index][1],
    z: projection.coordinates[index][2],
    kind: sample.kind,
    tokenId: sample.tokenId,
    rawToken: sample.rawToken,
    tokenCount: sample.tokenCount,
    chunkIndex: sample.chunkIndex,
    chunkCount: sample.chunkCount,
    tokenStart: sample.tokenStart,
    tokenEnd: sample.tokenEnd,
  }));

  onStatus({
    phase: "ready",
    message: `Model loaded · ${points.length} embeddings · ${reduction} projected`,
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
      loadTransformers().then(({ pipeline }) =>
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
      ),
    );
  }

  return extractorCache.get(modelId)!;
}

async function getImageExtractor(modelId: string, onStatus: (status: PipelineStatus) => void) {
  if (!imageExtractorCache.has(modelId)) {
    imageExtractorCache.set(
      modelId,
      loadTransformers().then(({ pipeline }) =>
        pipeline("image-feature-extraction", modelId, {
          dtype: "q8",
          progress_callback: (progress: ProgressInfo) => reportProgress(progress, onStatus, 0.5),
        }) as Promise<ImageFeatureExtractor>,
      ),
    );
  }

  return imageExtractorCache.get(modelId)!;
}

async function getTokenizer(modelId: string, onStatus: (status: PipelineStatus) => void) {
  if (!tokenizerCache.has(modelId)) {
    tokenizerCache.set(
      modelId,
      loadTransformers().then(
        ({ AutoTokenizer }) =>
          AutoTokenizer.from_pretrained(modelId, {
            progress_callback: (progress: ProgressInfo) => reportProgress(progress, onStatus, 0.26),
          }) as Promise<Tokenizer>,
      ),
    );
  }

  return tokenizerCache.get(modelId)!;
}

async function getCausalLm(modelId: string, onStatus: (status: PipelineStatus) => void) {
  if (!causalLmCache.has(modelId)) {
    causalLmCache.set(
      modelId,
      loadTransformers().then(({ AutoModelForCausalLM, AutoTokenizer }) =>
        Promise.all([
          AutoTokenizer.from_pretrained(modelId, {
            progress_callback: (progress: ProgressInfo) => reportProgress(progress, onStatus, 0.28),
          }) as Promise<Tokenizer>,
          AutoModelForCausalLM.from_pretrained(modelId, {
            dtype: "q4",
            progress_callback: (progress: ProgressInfo) => reportProgress(progress, onStatus, 0.52),
          }) as Promise<CausalLmModel>,
        ]).then(([tokenizer, model]) => ({ tokenizer, model })),
      ),
    );
  }

  return causalLmCache.get(modelId)!;
}

async function getClipTextModel(modelId: string, onStatus: (status: PipelineStatus) => void) {
  if (!clipTextCache.has(modelId)) {
    clipTextCache.set(
      modelId,
      loadTransformers().then(({ AutoTokenizer, CLIPTextModelWithProjection }) =>
        Promise.all([
          AutoTokenizer.from_pretrained(modelId, {
            progress_callback: (progress: ProgressInfo) => reportProgress(progress, onStatus, 0.28),
          }) as Promise<Tokenizer>,
          CLIPTextModelWithProjection.from_pretrained(modelId, {
            dtype: "q8",
            progress_callback: (progress: ProgressInfo) => reportProgress(progress, onStatus, 0.52),
          }) as Promise<ClipTextModel>,
        ]).then(([tokenizer, model]) => ({ tokenizer, model })),
      ),
    );
  }

  return clipTextCache.get(modelId)!;
}

async function getClipVisionModel(modelId: string, onStatus: (status: PipelineStatus) => void) {
  if (!clipVisionCache.has(modelId)) {
    clipVisionCache.set(
      modelId,
      loadTransformers().then(({ AutoProcessor, CLIPVisionModelWithProjection, RawImage }) =>
        Promise.all([
          AutoProcessor.from_pretrained(modelId, {
            progress_callback: (progress: ProgressInfo) => reportProgress(progress, onStatus, 0.28),
          }) as Promise<ImageProcessor>,
          CLIPVisionModelWithProjection.from_pretrained(modelId, {
            dtype: "q8",
            progress_callback: (progress: ProgressInfo) => reportProgress(progress, onStatus, 0.52),
          }) as Promise<ClipVisionModel>,
        ]).then(([processor, model]) => ({ processor, model, RawImage })),
      ),
    );
  }

  return clipVisionCache.get(modelId)!;
}

export async function buildEmbeddingInputPlan({
  model,
  inputType,
  snippets,
  files,
  onStatus,
}: {
  model: ModelPreset;
  inputType: Exclude<InputType, "tokens">;
  snippets: TextSnippet[];
  files: File[];
  onStatus: (status: PipelineStatus) => void;
}): Promise<EmbeddingInputPlan> {
  validateCompatibility(model, inputType, model.recommendedOutput);
  validateFiles(model, inputType, files);

  onStatus({ phase: "loading", message: "Counting tokens", progress: 0.04 });
  const tokenizer = await getTokenizer(model.id, onStatus);
  const chunkSize = effectiveChunkSize(tokenizer, model);
  const rawInputs = inputType === "files" ? await fileInputs(files) : [...snippetInputs(snippets), ...(await fileInputs(files))];
  const items: InputPlanItem[] = [];
  const samples: PlannedEmbeddingSample[] = [];

  for (const input of rawInputs) {
    const normalized = normalizeText(input.text);
    if (!normalized) {
      items.push({
        id: input.id,
        label: input.label,
        source: input.source,
        tokenCount: 0,
        chunkCount: 0,
        status: "skipped" as const,
        message: "Skipped empty input",
      });
      continue;
    }

    const tokenIds = tokenizer.encode(normalized, { add_special_tokens: false });
    const chunks = chunkTokenIds(tokenIds, chunkSize);
    const chunkCount = chunks.length;
    const status = chunkCount > 1 ? "chunked" : "ready";
    items.push({
      id: input.id,
      label: input.label,
      source: input.source,
      tokenCount: tokenIds.length,
      chunkCount,
      status,
      message: status === "chunked" ? `${chunkCount.toLocaleString()} chunks with ${CHUNK_OVERLAP_TOKENS} token overlap` : "Fits in one model call",
    });

    chunks.forEach((chunk, index) => {
      const chunkText = tokenizer.decode(chunk.ids, { skip_special_tokens: true }).trim() || normalized;
      samples.push({
        id: `${input.id}-${index}`,
        text: chunkText,
        label: input.label,
        parentLabel: input.label,
        source: input.source,
        kind: "input",
        tokenCount: chunk.ids.length,
        chunkIndex: index + 1,
        chunkCount,
        tokenStart: chunk.start + 1,
        tokenEnd: chunk.end,
      });
    });
  }

  const totalTokens = items.reduce((sum, item) => sum + item.tokenCount, 0);
  const totalChunks = items.reduce((sum, item) => sum + item.chunkCount, 0);
  const skippedCount = items.filter((item) => item.status === "skipped").length;
  onStatus({
    phase: "ready",
    message: `${totalTokens.toLocaleString()} tokens · ${totalChunks.toLocaleString()} chunks`,
    progress: 1,
  });

  return {
    inputType,
    modelId: model.id,
    chunkSize,
    overlapTokens: CHUNK_OVERLAP_TOKENS,
    items,
    samples,
    totalTokens,
    totalChunks,
    skippedCount,
  };
}

function resolvePlannedSamples(inputPlan?: EmbeddingInputPlan | null): ResolvedSample[] {
  if (!inputPlan) {
    throw new Error("Token count is still being prepared. Wait for the input plan before running.");
  }

  return inputPlan.samples.map((sample) => ({
    text: sample.text,
    label: sample.label,
    parentLabel: sample.parentLabel,
    source: sample.source,
    kind: sample.kind,
    tokenCount: sample.tokenCount,
    chunkIndex: sample.chunkIndex,
    chunkCount: sample.chunkCount,
    tokenStart: sample.tokenStart,
    tokenEnd: sample.tokenEnd,
  }));
}

async function extractVectors(
  model: ModelPreset,
  samples: ResolvedSample[],
  inputType: InputType,
  outputMode: OutputMode,
  onStatus: (status: PipelineStatus) => void,
) {
  if (typeof Worker === "undefined" || inputType === "tokens") {
    return extractVectorsCore(model, samples, inputType, outputMode, onStatus);
  }

  try {
    return await extractVectorsInWorker(model, samples, inputType, outputMode, onStatus);
  } catch (error) {
    console.warn("Embedding worker failed, falling back to main thread extraction.", error);
    if (model.task === "image-feature-extraction") {
      throw error;
    }
    return extractVectorsCore(model, samples, inputType, outputMode, onStatus);
  }
}

async function extractVectorsInWorker(
  model: ModelPreset,
  samples: ResolvedSample[],
  inputType: InputType,
  outputMode: OutputMode,
  onStatus: (status: PipelineStatus) => void,
) {
  const id = crypto.randomUUID();
  const worker = await createEmbeddingWorker();

  return new Promise<Float32Array[]>((resolve, reject) => {
    worker.onmessage = (event: MessageEvent<EmbeddingWorkerMessage>) => {
      const message = event.data;
      if (message.id !== id) return;

      if (message.type === "status") {
        onStatus(message.status);
        return;
      }

      worker.terminate();
      if (message.type === "error") {
        reject(new Error(message.message));
        return;
      }

      resolve(unpackVectors(new Float32Array(message.buffer), message.rows, message.dimensions));
    };

    worker.onerror = (event) => {
      worker.terminate();
      reject(new Error(event.message || "Embedding worker failed"));
    };

    worker.postMessage({ id, model, samples, inputType, outputMode });
  });
}

async function createEmbeddingWorker() {
  const { default: EmbeddingWorker } = await import("./embeddings.worker?worker");
  return new EmbeddingWorker();
}

function unpackVectors(data: Float32Array, rows: number, dimensions: number) {
  return Array.from({ length: rows }, (_, index) => data.subarray(index * dimensions, (index + 1) * dimensions));
}

export async function extractVectorsCore(
  model: ModelPreset,
  samples: ResolvedSample[],
  inputType: InputType,
  outputMode: OutputMode,
  onStatus: (status: PipelineStatus) => void,
) {
  onStatus({ phase: "loading", message: "Loading ONNX model", progress: 0.05 });

  if (inputType === "tokens") {
    onStatus({ phase: "embedding", message: `Projecting ${samples.length.toLocaleString()} tokenizer tokens`, progress: 0.64 });
    return tokenizerFeatureVectors(samples);
  }

  if (model.task === "text-generation") {
    const lm = await getCausalLm(model.id, onStatus);
    onStatus({ phase: "embedding", message: "Pooling language-model layer states", progress: 0.62 });
    return extractCausalLmVectors(lm, samples.map((sample) => sample.text), outputMode);
  }

  if (model.task === "clip-text") {
    return extractClipVectors(model.id, samples, onStatus);
  }

  if (model.task === "image-feature-extraction") {
    const imageFiles = samples.map((sample) => sample.image).filter((file): file is File => Boolean(file));
    if (imageFiles.length !== samples.length) {
      throw new Error("Image embedding requires image files.");
    }

    const extractor = await getImageExtractor(model.id, onStatus);
    onStatus({ phase: "embedding", message: `Extracting ${imageFiles.length.toLocaleString()} image embeddings`, progress: 0.62 });
    const output = await extractor(imageFiles);
    return normalizeRows(tensorRows(output.tolist()));
  }

  const extractor = await getExtractor(model.id, onStatus);

  if (outputMode === "tokens") {
    const output = await extractor(samples.map((sample) => sample.text), { pooling: "mean", normalize: true });
    return tensorRows(output.tolist());
  }

  const pooling = outputMode === "hidden-4" ? "cls" : "mean";
  const output = await extractor(samples.map((sample) => sample.text), { pooling, normalize: true });
  return tensorRows(output.tolist());
}

async function extractCausalLmVectors(bundle: CausalLmBundle, texts: string[], outputMode: OutputMode) {
  const inputs = bundle.tokenizer(texts, {
    padding: true,
    truncation: true,
    max_length: 96,
  });
  const outputs = await bundle.model(inputs);
  const layer = outputMode === "final" ? inferLastLayer(outputs) : layerFromOutputMode(outputMode);
  if (layer === null) {
    throw new Error(`SmolLM2 cannot use ${outputLabel(outputMode, "text-generation").toLowerCase()} as a layer state.`);
  }
  const valueState = outputs[`present.${layer}.value`];

  if (!isTensorLike(valueState) || valueState.dims.length !== 4) {
    throw new Error(`SmolLM2 did not return value states for layer ${layer}.`);
  }

  return normalizeRows(poolValueStates(valueState, inputs.attention_mask));
}

async function extractClipTextVectors(bundle: ClipTextBundle, texts: string[]) {
  const inputs = bundle.tokenizer(texts, {
    padding: true,
    truncation: true,
    max_length: 77,
  });
  const outputs = await bundle.model(inputs);
  const textEmbeds = outputs.text_embeds;

  if (!isTensorLike(textEmbeds) || textEmbeds.dims.length !== 2) {
    throw new Error("CLIP did not return text embeddings.");
  }

  return normalizeRows(tensorDataRows(textEmbeds));
}

async function extractClipImageVectors(bundle: ClipVisionBundle, files: File[]) {
  const images = await Promise.all(files.map((file) => bundle.RawImage.fromBlob(file)));
  const inputs = await bundle.processor(images);
  const outputs = await bundle.model(inputs);
  const imageEmbeds = outputs.image_embeds;

  if (!isTensorLike(imageEmbeds) || imageEmbeds.dims.length !== 2) {
    throw new Error("CLIP did not return image embeddings.");
  }

  return normalizeRows(tensorDataRows(imageEmbeds));
}

async function extractClipVectors(modelId: string, samples: ResolvedSample[], onStatus: (status: PipelineStatus) => void) {
  const vectors: Array<number[] | undefined> = Array.from({ length: samples.length });
  const textSamples = samples
    .map((sample, index) => ({ sample, index }))
    .filter(({ sample }) => sample.kind !== "image");
  const imageSamples = samples
    .map((sample, index) => ({ sample, index }))
    .filter(({ sample }) => sample.kind === "image" && sample.image);

  if (textSamples.length > 0) {
    const clip = await getClipTextModel(modelId, onStatus);
    onStatus({ phase: "embedding", message: "Computing CLIP text embeddings", progress: imageSamples.length ? 0.58 : 0.62 });
    const textVectors = await extractClipTextVectors(
      clip,
      textSamples.map(({ sample }) => sample.text),
    );
    textSamples.forEach(({ index }, vectorIndex) => {
      vectors[index] = textVectors[vectorIndex];
    });
  }

  if (imageSamples.length > 0) {
    const clip = await getClipVisionModel(modelId, onStatus);
    onStatus({ phase: "embedding", message: "Computing CLIP image embeddings", progress: textSamples.length ? 0.68 : 0.62 });
    const imageVectors = await extractClipImageVectors(
      clip,
      imageSamples.map(({ sample }) => sample.image!),
    );
    imageSamples.forEach(({ index }, vectorIndex) => {
      vectors[index] = imageVectors[vectorIndex];
    });
  }

  if (vectors.some((vector) => !vector)) {
    throw new Error("CLIP did not return embeddings for every input.");
  }

  return vectors as number[][];
}

async function resolveSamples(
  inputType: InputType,
  snippets: TextSnippet[],
  files: File[],
  model: ModelPreset,
  onStatus: (status: PipelineStatus) => void,
): Promise<ResolvedSample[]> {
  if (inputType === "tokens") {
    onStatus({ phase: "loading", message: "Loading tokenizer vocabulary", progress: 0.04 });
    const tokenizer = await getTokenizer(model.id, onStatus);
    return resolveTokenSamples(tokenizer.get_vocab()).map(([token, tokenId]) => ({
      text: token,
      label: displayToken(token),
      source: `Token id ${tokenId}`,
      kind: "token",
      tokenId,
      rawToken: token,
    }));
  }

  if (inputType === "files") {
    if (model.task === "image-feature-extraction") {
      return files.map((file) => ({
        text: imageDescription(file),
        label: file.name,
        source: file.name,
        kind: "image" as const,
        image: file,
      }));
    }

    return fileSamples(files, model);
  }

  const loadedFiles = await fileSamples(files, model);
  return [
    ...snippets
      .filter((snippet) => snippet.text.trim().length > 0)
      .map((snippet) => ({
        text: snippet.text.trim(),
        label: snippet.text.trim(),
        source: "Typed text",
        kind: "input" as const,
      })),
    ...loadedFiles,
  ];
}

function snippetInputs(snippets: TextSnippet[]) {
  return snippets.map((snippet, index) => {
    const text = normalizeText(snippet.text);
    return {
      id: snippet.id,
      text,
      label: text ? trimText(text, 42) : `Snippet ${index + 1}`,
      source: "Typed text",
    };
  });
}

async function fileInputs(files: File[]) {
  return Promise.all(
    files.map(async (file) => ({
      id: `${file.name}-${file.size}-${file.lastModified}`,
      text: await file.text(),
      label: file.name,
      source: file.name,
    })),
  );
}

async function fileSamples(files: File[], model: ModelPreset): Promise<ResolvedSample[]> {
  const samples = await Promise.all(
    files.map(async (file): Promise<ResolvedSample | null> => {
      const kind = classifyFile(file);
      if (kind === "image" && model.supportsImages) {
        return {
          text: imageDescription(file),
          label: file.name,
          source: file.name,
          kind: "image",
          image: file,
        };
      }

      if (kind === "text" && model.task !== "image-feature-extraction") {
        const text = trimText(await file.text(), 260);
        if (!text) return null;
        return {
          text,
          label: file.name,
          source: file.name,
          kind: "input",
        };
      }

      return null;
    }),
  );

  return samples.filter((sample): sample is ResolvedSample => Boolean(sample));
}

function normalizeText(value: string) {
  return value.replace(/\s+/g, " ").trim();
}

function effectiveChunkSize(tokenizer: Tokenizer, model: ModelPreset) {
  const specialTokenCount = tokenizer.encode("", { add_special_tokens: true }).length;
  return Math.max(8, model.maxInputTokens - specialTokenCount);
}

function chunkTokenIds(tokenIds: number[], chunkSize: number) {
  if (tokenIds.length === 0) {
    return [];
  }

  const overlap = Math.min(CHUNK_OVERLAP_TOKENS, Math.max(0, chunkSize - 1));
  const stride = Math.max(1, chunkSize - overlap);
  const chunks: Array<{ ids: number[]; start: number; end: number }> = [];

  for (let start = 0; start < tokenIds.length; start += stride) {
    const end = Math.min(start + chunkSize, tokenIds.length);
    chunks.push({
      ids: tokenIds.slice(start, end),
      start,
      end,
    });
    if (end === tokenIds.length) break;
  }

  return chunks;
}

function displayToken(token: string) {
  if (token === " ") return "space";
  if (token === "\n") return "newline";
  return token.replaceAll("Ġ", " ").replaceAll("▁", " ").replaceAll("</w>", "");
}

function resolveTokenSamples(vocabulary: Map<string, number>) {
  const entries = Array.from(vocabulary.entries()).sort((left, right) => left[1] - right[1]);
  const visibleTokens = entries.filter(([token]) => !isSpecialToken(token) && displayToken(token).trim().length > 0);
  if (visibleTokens.length >= 2) {
    return visibleTokens.slice(0, MAX_TOKEN_POINTS);
  }

  return entries.slice(0, 2);
}

function isSpecialToken(token: string) {
  return /^<.*>$/.test(token) || /^\[.*\]$/.test(token);
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

function tensorDataRows(tensor: { data: ArrayLike<number>; dims: number[] }) {
  const [rows, dimensions] = tensor.dims;
  return Array.from({ length: rows }, (_, rowIndex) => {
    const offset = rowIndex * dimensions;
    return Array.from({ length: dimensions }, (_, col) => Number(tensor.data[offset + col] ?? 0));
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

function validateCompatibility(model: ModelPreset, inputType: InputType, outputMode: OutputMode) {
  if (!model.outputModes.includes(outputMode)) {
    throw new Error(`${model.label} does not expose ${outputLabel(outputMode, model.task).toLowerCase()}.`);
  }

  if (inputType === "tokens" && !model.outputModes.includes("tokens")) {
    throw new Error(`${model.label} does not expose token embedding layers.`);
  }

  if (model.task === "image-feature-extraction" && inputType !== "files") {
    throw new Error(`${model.label} embeds image files. Switch the input type to files before running.`);
  }
}

function validateFiles(model: ModelPreset, inputType: InputType, files: File[]) {
  if (inputType !== "files") return;

  const incompatibleFiles = files.filter((file) => !isFileCompatibleWithModel(file, model));
  if (incompatibleFiles.length > 0) {
    throw new Error(`${model.label} cannot embed ${incompatibleFiles[0].type || incompatibleFiles[0].name}.`);
  }
}

function outputLabel(outputMode: OutputMode, task?: ModelPreset["task"]) {
  if (task === "clip-text") return "CLIP embedding";
  if (task === "text-generation") {
    if (outputMode === "final") return "Final LM value state";
    if (outputMode === "tokens") return "Tokenizer subword features";
    const layer = layerFromOutputMode(outputMode);
    return layer === null ? "LM value state" : `Layer ${layer} · LM value state`;
  }
  const layer = layerFromOutputMode(outputMode);
  if (layer !== null) return `Layer ${layer} · hidden state`;
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

function layerFromOutputMode(outputMode: OutputMode) {
  const match = outputMode.match(/^hidden-(\d+)$/);
  return match ? Number(match[1]) : null;
}

function isTensorLike(value: unknown): value is { data: ArrayLike<number>; dims: number[] } {
  return (
    typeof value === "object" &&
    value !== null &&
    "data" in value &&
    "dims" in value &&
    Array.isArray((value as { dims?: unknown }).dims)
  );
}

function classifyFileMime(mimeType: string): "text" | "image" | "unsupported" {
  if (mimeType.startsWith("image/")) return "image";
  if (mimeType.startsWith("text/")) return "text";
  if (["application/json", "application/csv", "application/xml", "application/x-ndjson"].includes(mimeType)) return "text";
  return "unsupported";
}

function classifyFile(file: File) {
  const mimeKind = classifyFileMime(file.type);
  if (mimeKind !== "unsupported") return mimeKind;
  if (/\.(txt|md|csv|json|jsonl|ndjson|xml)$/i.test(file.name)) return "text";
  return "unsupported";
}

function isFileCompatibleWithModel(file: File, model: ModelPreset) {
  const kind = classifyFile(file);
  if (kind === "image") return model.supportsImages;
  if (kind === "text") return model.task !== "image-feature-extraction";
  return false;
}

function tokenizerFeatureVectors(samples: ResolvedSample[]) {
  return normalizeRows(
    samples.map((sample) => {
      const token = sample.text;
      const id = sample.tokenId ?? 0;
      const chars = Array.from(token);
      const vector = Array.from({ length: 32 }, () => 0);
      vector[0] = id / Math.max(samples.length - 1, 1);
      vector[1] = Math.log1p(chars.length) / 4;
      vector[2] = token.startsWith("Ġ") || token.startsWith("▁") ? 1 : 0;
      vector[3] = token.startsWith("##") ? 1 : 0;
      vector[4] = /^\W+$/.test(token) ? 1 : 0;
      vector[5] = /^\d+$/.test(token) ? 1 : 0;
      vector[6] = token.includes("<") || token.includes("[") ? 1 : 0;

      for (let i = 0; i < chars.length; i += 1) {
        const code = chars[i].codePointAt(0) ?? 0;
        vector[7 + (i % 25)] += ((code % 97) / 97) * (1 / Math.sqrt(i + 1));
      }

      return vector;
    }),
  );
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

function imageDescription(file: File) {
  const size =
    file.size >= 1024 * 1024 ? `${(file.size / (1024 * 1024)).toFixed(1)} MB` : `${Math.max(1, Math.round(file.size / 1024))} KB`;
  return `${file.type || "image file"} · ${size}`;
}

export function runColor(index: number) {
  return ["#2563eb", "#f59e0b", "#f43f72", "#14b8a6", "#7c3aed"][index % 5];
}

export const __testing = {
  chunkTokenIds,
  displayToken,
  layerFromOutputMode,
  outputLabel,
  resolveTokenSamples,
  validateFiles,
};
