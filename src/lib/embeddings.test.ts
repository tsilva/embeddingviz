import { beforeEach, describe, expect, it, vi } from "vitest";
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
  AutoTokenizer: {
    from_pretrained: vi.fn(),
  },
  AutoModelForCausalLM: {
    from_pretrained: vi.fn(),
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
});
