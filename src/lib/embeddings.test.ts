import { beforeEach, describe, expect, it, vi } from "vitest";
import { MODEL_PRESETS } from "../data";
import type { ModelPreset, PlannedEmbeddingSample } from "../types";
import { __testing, createEmbeddingRun } from "./embeddings";
import { projectReduction } from "./reductions";

vi.mock("@huggingface/transformers", () => ({
  env: {},
  pipeline: vi.fn(async () => async () => ({
    tolist: () => [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ],
  })),
  AutoProcessor: {
    from_pretrained: vi.fn(async () => async (images: unknown[] | unknown) => ({
      __images: Array.isArray(images) ? images : [images],
    })),
  },
  AutoTokenizer: {
    from_pretrained: vi.fn(async () => {
      const tokenizer = ((texts: string[]) => ({ __texts: texts })) as {
        (texts: string[]): { __texts: string[] };
        encode: (text: string) => number[];
        decode: (ids: number[]) => string;
        get_vocab: () => Map<string, number>;
      };
      tokenizer.encode = (text: string) => text.split(/\s+/).filter(Boolean).map((_, index) => index + 1);
      tokenizer.decode = (ids: number[]) => ids.map((id) => `token-${id}`).join(" ");
      tokenizer.get_vocab = () => new Map();
      return tokenizer;
    }),
  },
  AutoModelForCausalLM: {
    from_pretrained: vi.fn(),
  },
  CLIPTextModelWithProjection: {
    from_pretrained: vi.fn(async () => async (inputs: { __texts: string[] }) => {
      const rows = inputs.__texts.map((_, index) => (index === 0 ? [1, 0, 0] : [0, 0, 1]));
      return {
        text_embeds: {
          dims: [rows.length, 3],
          data: rows.flat(),
        },
      };
    }),
  },
  CLIPVisionModelWithProjection: {
    from_pretrained: vi.fn(async () => async (inputs: { __images: unknown[] }) => {
      const rows = inputs.__images.map(() => [0, 1, 0]);
      return {
        image_embeds: {
          dims: [rows.length, 3],
          data: rows.flat(),
        },
      };
    }),
  },
  RawImage: {
    fromBlob: vi.fn(async (blob: Blob) => blob),
  },
}));

vi.mock("./reductions", () => ({
  projectReduction: vi.fn(async () => ({
    coordinates: [
      [-5, 0, 0],
      [0, 5, 0],
      [5, -5, 0],
    ],
    explained: [0.7, 0.2, 0.1],
  })),
}));

const textModel: ModelPreset = {
  id: "test/text-model",
  label: "Text Model",
  task: "feature-extraction",
  summary: "Text test model",
  recommendedOutput: "final",
  outputModes: ["final"],
  maxInputTokens: 128,
  supportsImages: false,
  note: "Test fixture",
};

const clipModel: ModelPreset = {
  id: "test/clip-model",
  label: "CLIP Model",
  task: "clip-text",
  summary: "CLIP test model",
  recommendedOutput: "final",
  outputModes: ["final"],
  maxInputTokens: 77,
  supportsImages: true,
  note: "Test fixture",
};

describe("embedding internals", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("chunks token ids with bounded overlap and complete coverage", () => {
    const tokenIds = Array.from({ length: 80 }, (_, index) => index + 1);
    const chunks = __testing.chunkTokenIds(tokenIds, 32);

    expect(chunks[0]).toEqual({ ids: tokenIds.slice(0, 32), start: 0, end: 32 });
    expect(chunks[1]).toEqual({ ids: tokenIds.slice(8, 40), start: 8, end: 40 });
    expect(chunks.at(-1)).toEqual({ ids: tokenIds.slice(48, 80), start: 48, end: 80 });
  });

  it("filters special and blank tokenizer samples while preserving token id order", () => {
    const samples = __testing.resolveTokenSamples(
      new Map([
        ["<s>", 0],
        ["[PAD]", 1],
        ["Ġ", 2],
        ["Ġweather", 3],
        ["model", 4],
        ["▁vector", 5],
      ]),
    );

    expect(samples).toEqual([
      ["Ġweather", 3],
      ["model", 4],
      ["▁vector", 5],
    ]);
  });

  it("lists SmolLM2 layer outputs from final to first layer", () => {
    const model = MODEL_PRESETS.find((preset) => preset.id === "HuggingFaceTB/SmolLM2-135M-Instruct");

    expect(model?.recommendedOutput).toBe("final");
    expect(model?.outputModes).toEqual([
      "final",
      ...Array.from({ length: 30 }, (_, index) => `hidden-${29 - index}`),
      "tokens",
    ]);
    expect(__testing.outputLabel("hidden-29", "text-generation")).toBe("Layer 29 · LM value state");
    expect(__testing.outputLabel("hidden-0", "text-generation")).toBe("Layer 0 · LM value state");
    expect(__testing.layerFromOutputMode("hidden-12")).toBe(12);
  });

  it("rejects files incompatible with the selected model", () => {
    const imageFile = new File(["png"], "plot.png", { type: "image/png" });
    const markdownFile = new File(["# notes"], "notes.md", { type: "" });

    expect(() => __testing.validateFiles(textModel, "files", [markdownFile])).not.toThrow();
    expect(() => __testing.validateFiles(textModel, "files", [imageFile])).toThrow("Text Model cannot embed image/png");
  });

  it("creates a projected run from planned text samples", async () => {
    const samples: PlannedEmbeddingSample[] = ["alpha", "beta", "gamma"].map((label, index) => ({
      id: `sample-${index}`,
      text: `${label} text`,
      label,
      parentLabel: label,
      source: "Text snippets",
      kind: "input",
      tokenCount: 2,
      chunkIndex: 1,
      chunkCount: 1,
      tokenStart: 1,
      tokenEnd: 2,
    }));
    const statuses: string[] = [];

    const result = await createEmbeddingRun({
      model: textModel,
      outputMode: "final",
      inputType: "text",
      reduction: "PCA",
      snippets: [],
      files: [],
      inputPlan: {
        inputType: "text",
        modelId: textModel.id,
        chunkSize: 120,
        overlapTokens: 24,
        items: [],
        samples,
        totalTokens: 6,
        totalChunks: 3,
        skippedCount: 0,
      },
      onStatus: (status) => statuses.push(status.phase),
    });

    expect(projectReduction).toHaveBeenCalledWith(
      [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
      "PCA",
      expect.any(Function),
    );
    expect(result.explained).toEqual([0.7, 0.2, 0.1]);
    expect(result.points.map((point) => [point.label, point.x, point.y])).toEqual([
      ["alpha", -5, 0],
      ["beta", 0, 5],
      ["gamma", 5, -5],
    ]);
    expect(statuses).toContain("ready");
  });

  it("routes CLIP text and image inputs to their matching encoders while preserving order", async () => {
    const imageFile = new File(["png"], "plot.png", { type: "image/png" });
    const textFile = new File(["delta notes"], "notes.md", { type: "text/markdown" });

    await createEmbeddingRun({
      model: clipModel,
      outputMode: "final",
      inputType: "text",
      reduction: "PCA",
      snippets: [{ id: "s1", text: "alpha prompt" }],
      files: [imageFile, textFile],
      inputPlan: null,
      onStatus: () => {},
    });

    expect(projectReduction).toHaveBeenLastCalledWith(
      [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
      "PCA",
      expect.any(Function),
    );
  });
});
