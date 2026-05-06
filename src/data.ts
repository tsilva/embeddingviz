import type { ModelPreset, TextSnippet } from "./types";

const SMOLLM2_LAYER_COUNT = 30;
const SMOLLM2_LAYER_OUTPUTS = Array.from({ length: SMOLLM2_LAYER_COUNT }, (_, index) => `hidden-${SMOLLM2_LAYER_COUNT - 1 - index}` as const);

export const MODEL_PRESETS: ModelPreset[] = [
  {
    id: "Xenova/paraphrase-MiniLM-L3-v2",
    label: "paraphrase-MiniLM-L3-v2",
    task: "feature-extraction",
    summary: "Text embedding model · ONNX ready",
    recommendedOutput: "final",
    outputModes: ["final"],
    maxInputTokens: 128,
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
    maxInputTokens: 256,
    supportsImages: false,
    note: "Higher quality default embedding model; useful after the tiny smoke test passes.",
  },
  {
    id: "HuggingFaceTB/SmolLM2-135M-Instruct",
    label: "SmolLM2-135M-Instruct",
    task: "text-generation",
    summary: "Language model · hidden-state workflow",
    recommendedOutput: "final",
    outputModes: ["final", ...SMOLLM2_LAYER_OUTPUTS, "tokens"],
    maxInputTokens: 96,
    supportsImages: false,
    note: "Useful compatibility target for LM hidden-state exploration; final pooled embeddings are not the recommended default.",
  },
  {
    id: "Xenova/clip-vit-base-patch32",
    label: "CLIP ViT-B/32",
    task: "clip-text",
    summary: "CLIP text/image encoders · ONNX ready",
    recommendedOutput: "final",
    outputModes: ["final"],
    maxInputTokens: 77,
    supportsImages: true,
    note: "CLIP text and image towers for comparing prompts and images in the shared CLIP embedding space.",
  },
];

export const SAMPLE_SNIPPETS: TextSnippet[] = [
  {
    id: "s1",
    text: "Thunderstorms rolled across the harbor before sunrise.",
  },
  {
    id: "s2",
    text: "A cold front brought steady rain to the valley.",
  },
  {
    id: "s3",
    text: "Bright afternoon sunlight cleared the morning fog.",
  },
  {
    id: "s4",
    text: "Trail runners climbed the ridge above the pine forest.",
  },
  {
    id: "s5",
    text: "Backpackers followed a rocky path through alpine meadows.",
  },
  {
    id: "s6",
    text: "A mountain guide checked the map beside the campsite.",
  },
  {
    id: "s7",
    text: "Neural networks learn useful patterns from examples.",
  },
  {
    id: "s8",
    text: "Transformers convert tokens into contextual vectors.",
  },
  {
    id: "s9",
    text: "Embedding spaces reveal semantic neighborhoods.",
  },
  {
    id: "s10",
    text: "The chef simmered tomato sauce with basil and garlic.",
  },
  {
    id: "s11",
    text: "Fresh bread cooled on the kitchen counter after baking.",
  },
  {
    id: "s12",
    text: "The recipe called for olive oil, lemon, and herbs.",
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
