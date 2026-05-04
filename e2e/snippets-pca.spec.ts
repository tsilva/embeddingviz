import { expect, test } from "@playwright/test";

test("enter snippets, run PCA, select a point", async ({ page }) => {
  await page.goto("/?mockEmbeddings=1");

  const snippets = page.getByTestId("snippet-input");
  await expect(snippets).toHaveCount(3);
  await snippets.nth(0).fill("alpha forest");
  await snippets.nth(1).fill("beta weather");
  await snippets.nth(2).fill("gamma cliffs");

  await page.getByTestId("reduction-PCA").click();
  await page.getByTestId("run-projection").click();

  await expect(page.getByText("3 visible points")).toBeVisible();
  await expect(page.getByTestId("selected-point-label")).toHaveText("alpha forest");

  const canvas = page.getByTestId("point-cloud-canvas");
  const box = await canvas.boundingBox();
  expect(box).not.toBeNull();
  await canvas.click({
    position: {
      x: (box!.width * 740) / 860,
      y: (box!.height * 538) / 640,
    },
  });

  await expect(page.getByTestId("selected-point-label")).toHaveText("gamma cliffs");
});

test("input widget shows token plan totals and per-snippet metadata", async ({ page }) => {
  await page.goto("/?mockEmbeddings=1");

  const snippets = page.getByTestId("snippet-input");
  await snippets.nth(0).fill("alpha forest");
  await snippets.nth(1).fill("beta weather");
  await snippets.nth(2).fill("gamma cliffs");

  await expect(page.getByTestId("token-plan-overview")).toContainText("Chunks appear after Run");
  await page.getByTestId("run-projection").click();

  await expect(page.getByTestId("token-plan-overview")).toContainText("9 tokens");
  await expect(page.getByTestId("token-plan-overview")).toContainText("3 chunks");
  await expect(page.getByTestId("token-plan-overview")).toContainText("24 token overlap");

  const itemMetadata = page.getByTestId("input-item-meta");
  await expect(itemMetadata).toHaveCount(3);
  await expect(itemMetadata.nth(0)).toContainText("3 tokens");
  await expect(itemMetadata.nth(0)).toContainText("1 chunk");
});

test("selected tooltip label truncates long point labels", async ({ page }) => {
  const longLabel = "alpha forest with a very long selected point label that should not spill out of the tooltip";

  await page.goto("/?mockEmbeddings=1");

  const snippets = page.getByTestId("snippet-input");
  await expect(snippets).toHaveCount(3);
  await snippets.nth(0).fill(longLabel);
  await snippets.nth(1).fill("beta weather");
  await snippets.nth(2).fill("gamma cliffs");

  await page.getByTestId("reduction-PCA").click();
  await page.getByTestId("run-projection").click();

  const tooltipLabel = page.getByTestId("selected-point-tooltip-label");
  await expect(tooltipLabel).toHaveAttribute("title", longLabel);
  await expect(tooltipLabel).toHaveCSS("overflow", "hidden");
  await expect(tooltipLabel).toHaveCSS("text-overflow", "ellipsis");
  await expect(tooltipLabel).toHaveCSS("white-space", "nowrap");

  const isClipped = await tooltipLabel.evaluate((element) => element.scrollWidth > element.clientWidth);
  expect(isClipped).toBe(true);
});

test("nearest point results are ordered by similarity across visible runs", async ({ page }) => {
  await page.addInitScript(() => {
    let runIndex = 0;
    const fixtures = [
      [
        { label: "previous close", vector: [0.99, 0.1], x: -4, y: 0 },
        { label: "previous mid", vector: [0.6, 0.8], x: 0, y: 4 },
        { label: "previous far", vector: [0, 1], x: 4, y: 0 },
      ],
      [
        { label: "current anchor", vector: [1, 0], x: -4, y: -4 },
        { label: "current mid", vector: [0.8, 0.6], x: 0, y: -1 },
        { label: "current far", vector: [0, 1], x: 4, y: -4 },
      ],
    ];

    window.__EMBEDDINGVIZ_TEST__ = {
      buildEmbeddingInputPlan: async ({ inputType, model, onStatus }) => {
        onStatus({ phase: "ready", message: "3 chunks", progress: 1 });
        return {
          inputType,
          modelId: model.id,
          chunkSize: model.maxInputTokens,
          overlapTokens: 24,
          items: [],
          samples: [],
          totalTokens: 9,
          totalChunks: 3,
          skippedCount: 0,
        };
      },
      createEmbeddingRun: async ({ onStatus }) => {
        const fixture = fixtures[Math.min(runIndex, fixtures.length - 1)];
        const currentRun = runIndex;
        runIndex += 1;
        onStatus({ phase: "embedding", message: "Mock embeddings", progress: 0.6 });
        onStatus({ phase: "projecting", message: "Mock projection", progress: 0.85 });
        onStatus({ phase: "ready", message: "Mock complete", progress: 1 });

        return {
          explained: [0.8, 0.15, 0.05],
          points: fixture.map((point, index) => ({
            id: `run-${currentRun}-point-${index}`,
            label: point.label,
            parentLabel: point.label,
            snippet: point.label,
            source: "Test fixture",
            output: "Final embedding",
            vector: point.vector,
            x: point.x,
            y: point.y,
            z: 0,
            kind: "input",
            tokenCount: 3,
            chunkIndex: 1,
            chunkCount: 1,
            tokenStart: 1,
            tokenEnd: 3,
          })),
        };
      },
      runColor: (index) => ["#2563eb", "#f59e0b"][index % 2],
    };
  });

  await page.goto("/?mockEmbeddings=1");

  const snippets = page.getByTestId("snippet-input");
  await snippets.nth(0).fill("alpha forest");
  await snippets.nth(1).fill("beta weather");
  await snippets.nth(2).fill("gamma cliffs");

  await page.getByTestId("run-projection").click();
  await expect(page.getByText("3 visible points")).toBeVisible();
  await page.getByTestId("run-projection").click();
  await expect(page.getByText("6 visible points")).toBeVisible();
  await expect(page.getByTestId("selected-point-label")).toHaveText("current anchor");

  const nearestLabels = page.getByTestId("nearest-row").locator("strong");
  await expect(nearestLabels.nth(0)).toHaveText("previous close");
  await expect(nearestLabels.nth(1)).toHaveText("current mid");
  await expect(nearestLabels.nth(2)).toHaveText("previous mid");
});

test("wheel zoom only captures scrolling while the left mouse button is down", async ({ page }) => {
  await page.goto("/?mockEmbeddings=1");

  const snippets = page.getByTestId("snippet-input");
  await snippets.nth(0).fill("alpha forest");
  await snippets.nth(1).fill("beta weather");
  await snippets.nth(2).fill("gamma cliffs");
  await page.getByTestId("run-projection").click();
  await expect(page.getByText("3 visible points")).toBeVisible();

  const canvas = page.getByTestId("point-cloud-canvas");
  const box = await canvas.boundingBox();
  expect(box).not.toBeNull();

  const alphaLabel = page.locator(".pointLabel", { hasText: "alpha forest" });
  const initialX = await alphaLabel.getAttribute("x");

  await canvas.hover({
    position: {
      x: box!.width / 2,
      y: box!.height / 2,
    },
  });
  await page.mouse.wheel(0, -200);
  await expect(alphaLabel).toHaveAttribute("x", initialX!);

  await page.mouse.move(box!.x + box!.width / 2, box!.y + box!.height / 2);
  await page.mouse.down({ button: "left" });
  await page.mouse.wheel(0, -200);
  await page.mouse.up({ button: "left" });

  await expect(alphaLabel).not.toHaveAttribute("x", initialX!);
});
