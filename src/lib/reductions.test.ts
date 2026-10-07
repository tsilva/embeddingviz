import { describe, expect, it, vi } from "vitest";
import { MAX_TSNE_POINTS } from "./reductionLimits";
import { projectReduction } from "./reductions";
import { projectReductionCore } from "./reductionCore";

describe("projection resource limits", () => {
  it("rejects vocabulary-sized t-SNE before launching a worker or reporting progress", async () => {
    const vectors = Array.from({ length: MAX_TSNE_POINTS + 1 }, () => [1, 2]);
    const onStatus = vi.fn();
    await expect(projectReduction(vectors, "t-SNE", onStatus)).rejects.toThrow("Choose PCA or UMAP");
    await expect(projectReductionCore(vectors, "t-SNE", onStatus)).rejects.toThrow("at most");
    expect(onStatus).not.toHaveBeenCalled();
  });

  it("does not restart a failed worker on the UI thread", async () => {
    const onStatus = vi.fn();
    class BrokenWorker {
      onerror?: (event: { message: string }) => void;
      postMessage() { this.onerror?.({ message: "Out of memory" }); }
      terminate() {}
    }
    vi.doMock("./reductions.worker?worker", () => ({ default: BrokenWorker }));
    vi.stubGlobal("Worker", BrokenWorker);
    try {
      await expect(projectReduction([[1, 0], [0, 1]], "PCA", onStatus)).rejects.toThrow("Out of memory");
      expect(onStatus).not.toHaveBeenCalled();
    } finally { vi.unstubAllGlobals(); vi.doUnmock("./reductions.worker?worker"); }
  });

  it("cancels a running projection by terminating its worker", async () => {
    const terminate = vi.fn();
    let start!: () => void;
    const started = new Promise<void>((resolve) => { start = resolve; });
    class PendingWorker {
      postMessage() { start(); }
      terminate = terminate;
    }
    vi.doMock("./reductions.worker?worker", () => ({ default: PendingWorker }));
    vi.stubGlobal("Worker", PendingWorker);
    try {
      const controller = new AbortController();
      const job = projectReduction([[1, 0], [0, 1]], "PCA", () => {}, controller.signal);
      const assertion = expect(job).rejects.toThrow("cancelled");
      await started;
      controller.abort();
      await assertion;
      expect(terminate).toHaveBeenCalledOnce();
    } finally { vi.unstubAllGlobals(); vi.doUnmock("./reductions.worker?worker"); }
  });
});
