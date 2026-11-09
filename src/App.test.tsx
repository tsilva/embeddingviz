import { act } from "react";
import { createRoot, hydrateRoot, type Root } from "react-dom/client";
import { renderToString } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { buildEmbeddingInputPlan, createEmbeddingRun } from "./lib/embeddings";

vi.mock("./lib/embeddings", () => ({ buildEmbeddingInputPlan: vi.fn(), createEmbeddingRun: vi.fn(), runColor: () => "#2563eb" }));

let root: Root;
let container: HTMLDivElement;
async function click(selector: string) {
  const button = container.querySelector<HTMLButtonElement>(selector);
  expect(button).not.toBeNull();
  await act(async () => { button!.click(); });
}
async function select(selector: string, value: string) {
  await act(async () => {
    const element = container.querySelector<HTMLSelectElement>(selector)!;
    element.value = value;
    element.dispatchEvent(new Event("change", { bubbles: true }));
  });
}

beforeEach(async () => {
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
  vi.mocked(buildEmbeddingInputPlan).mockImplementation(async ({ inputType, model, onStatus }) => {
    onStatus({ phase: "ready", message: "Plan complete", progress: 1 });
    return { inputType, modelId: model.id, chunkSize: 126, overlapTokens: 24, items: [], samples: [], totalTokens: 2, totalChunks: 2, skippedCount: 0 };
  });
  vi.mocked(createEmbeddingRun).mockImplementation(async ({ onStatus }) => {
    onStatus({ phase: "ready", message: "Run complete", progress: 1 });
    return { explained: [1, 0, 0], points: ["alpha", "beta"].map((label, index) => ({ id: label, label, snippet: label, source: "test", output: "Final embedding", vector: [index, 1], x: index, y: 0, z: 1 })) };
  });
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => { root.render(<App />); });
});

afterEach(async () => {
  await act(async () => { root.unmount(); });
  container.remove();
  vi.restoreAllMocks();
  vi.clearAllMocks();
});

describe("run workflow", () => {
  it("reuses the prerendered markup and runs a projection after hydration", async () => {
    await act(async () => { root.unmount(); });
    container.innerHTML = renderToString(<App />);
    const button = container.querySelector('[data-testid="run-projection"]');
    expect(button).not.toBeNull();
    const onRecoverableError = vi.fn();

    await act(async () => {
      root = hydrateRoot(container, <App />, { onRecoverableError });
    });

    expect(onRecoverableError).not.toHaveBeenCalled();
    expect(container.querySelector('[data-testid="run-projection"]')).toBe(button);
    await click('[data-testid="run-projection"]');
    expect(createEmbeddingRun).toHaveBeenCalledTimes(1);
  });

  it("initializes the canvas only when points exist and clears a filtered plot", async () => {
    const getContext = vi.mocked(HTMLCanvasElement.prototype.getContext);
    expect(getContext).not.toHaveBeenCalled();

    const context = {
      clearRect: vi.fn(),
      beginPath: vi.fn(),
      arc: vi.fn(),
      fill: vi.fn(),
      stroke: vi.fn(),
    };
    getContext.mockImplementation((type) =>
      type === "2d" ? context as unknown as CanvasRenderingContext2D : null,
    );

    await click('[data-testid="run-projection"]');
    expect(getContext).toHaveBeenCalledWith("webgl", { antialias: true });
    expect(context.arc).toHaveBeenCalled();
    context.clearRect.mockClear();
    context.arc.mockClear();

    const search = container.querySelector<HTMLInputElement>('[aria-label="Search points"]')!;
    const setInputValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!;
    await act(async () => {
      setInputValue.call(search, "no matching point");
      search.dispatchEvent(new Event("input", { bubbles: true }));
    });

    expect(container.querySelector(".emptyPlot")?.textContent).toContain("No points match");
    expect(context.clearRect).toHaveBeenCalledTimes(1);
    expect(context.arc).not.toHaveBeenCalled();
  });

  it("shows inference errors after a successful plan and recovers on retry", async () => {
    vi.mocked(createEmbeddingRun).mockRejectedValueOnce(new Error("Model download failed"));
    await click('[data-testid="run-projection"]');
    expect(container.querySelector('[role="alert"]')?.textContent).toContain("Model download failed");
    expect(container.querySelector<HTMLButtonElement>('[data-testid="run-projection"]')?.disabled).toBe(false);
    await click('[data-testid="run-projection"]');
    expect(container.querySelector('[role="alert"]')).toBeNull();
    expect(container.querySelectorAll(".runCard")).toHaveLength(1);
  });

  it("keeps the displayed PCA label when next-run UMAP is chosen", async () => {
    await click('[data-testid="run-projection"]');
    await click('[data-testid="reduction-UMAP"]');
    expect(container.querySelector(".plotCaption")?.textContent).toContain("PCA");
    expect(container.querySelector('[data-testid="reduction-UMAP"]')?.getAttribute("aria-pressed")).toBe("true");
    expect(createEmbeddingRun).toHaveBeenCalledTimes(1);
  });

  it("shows one independent projection and excludes other-model neighbors", async () => {
    await click('[data-testid="run-projection"]');
    await select("#model", "Xenova/clip-vit-base-patch32");
    await click('[data-testid="run-projection"]');
    expect(container.querySelectorAll('.runCard input:checked')).toHaveLength(1);
    expect(container.querySelectorAll('[data-testid="nearest-row"]')).toHaveLength(1);
    await click('.runCard:nth-of-type(2) input');
    expect(container.querySelector(".plotCaption")?.textContent).toContain("paraphrase-MiniLM-L3-v2");
    expect(container.querySelectorAll('.runCard input:checked')).toHaveLength(1);
  });

  it("switches plot and inspector together when a saved neighbor is selected", async () => {
    await click('[data-testid="run-projection"]');
    await click('[data-testid="run-projection"]');
    // The same source ID in a second run is distinct and a valid neighbor.
    expect(container.querySelectorAll('[data-testid="nearest-row"]')).toHaveLength(3);
    await click('[data-testid="nearest-row"]');
    expect(container.querySelector<HTMLInputElement>('.runCard:nth-of-type(2) input')?.checked).toBe(true);
    expect(container.querySelectorAll('.runCard input:checked')).toHaveLength(1);
  });

  it("disables 3D for a two-dimensional t-SNE result", async () => {
    vi.mocked(createEmbeddingRun).mockImplementationOnce(async ({ onStatus }) => {
      onStatus({ phase: "ready", message: "Complete", progress: 1 });
      return { explained: [0, 0, 0], points: Array.from({ length: 4 }, (_, index) => ({ id: `${index}`, label: `${index}`, snippet: "test", source: "test", output: "final", vector: [1, index], x: index, y: 0, z: 0 })) };
    });
    await click('[data-testid="reduction-t-SNE"]');
    await click('[data-testid="run-projection"]');
    const button = [...container.querySelectorAll<HTMLButtonElement>("button")].find((button) => button.textContent === "3D");
    expect(button?.disabled).toBe(true);
  });
});
