import type { ModelPreset, TextSnippet } from "./types";

export const MODEL_PRESETS: ModelPreset[] = [
  {
    id: "Xenova/paraphrase-MiniLM-L3-v2",
    label: "paraphrase-MiniLM-L3-v2",
    task: "feature-extraction",
    summary: "Text embedding model · ONNX ready",
    recommendedOutput: "final",
    outputModes: ["final"],
    supportsImages: false,
    note: "Very small encoder model; fast enough for browser-side MVP testing on Apple Silicon.",
  },
  {
    id: "onnx-community/all-MiniLM-L6-v2-ONNX",
    label: "all-MiniLM-L6-v2",
    task: "feature-extraction",
    summary: "Text embedding model · ONNX ready",
    recommendedOutput: "final",
    outputModes: ["final"],
    supportsImages: false,
    note: "Higher quality default embedding model; useful after the tiny smoke test passes.",
  },
  {
    id: "HuggingFaceTB/SmolLM2-135M-Instruct",
    label: "SmolLM2-135M-Instruct",
    task: "text-generation",
    summary: "Language model · hidden-state workflow",
    recommendedOutput: "hidden-4",
    outputModes: ["final", "hidden-4", "tokens"],
    supportsImages: false,
    note: "Useful compatibility target for LM hidden-state exploration; final pooled embeddings are not the recommended default.",
  },
  {
    id: "Xenova/clip-vit-base-patch32",
    label: "CLIP ViT-B/32",
    task: "image-feature-extraction",
    summary: "Vision model · image features",
    recommendedOutput: "final",
    outputModes: ["final"],
    supportsImages: true,
    note: "Image-capable model for future multimodal comparisons.",
  },
];

export const SAMPLE_SNIPPETS: TextSnippet[] = [
  {
    id: "s1",
    text: "The weather today is sunny and warm.",
  },
  {
    id: "s2",
    text: "I love hiking in the mountains.",
  },
  {
    id: "s3",
    text: "Machine learning models learn patterns.",
  },
  {
    id: "s4",
    text: "Transformers convert tokens into contextual vectors.",
  },
  {
    id: "s5",
    text: "A kitten watched birds from the windowsill.",
  },
  {
    id: "s6",
    text: "Rain clouds gathered above the coast.",
  },
  {
    id: "s7",
    text: "Trail runners crossed the forest ridge.",
  },
  {
    id: "s8",
    text: "Embedding spaces reveal semantic neighborhoods.",
  },
];

export const TOKEN_SAMPLE = [
  "transformers",
  "embedding",
  "model",
  "token",
  "vector",
  "semantic",
  "weather",
  "sunny",
  "warm",
  "rain",
  "mountains",
  "hiking",
  "trail",
  "outdoors",
  "cat",
  "kitten",
  "window",
  "bird",
];
