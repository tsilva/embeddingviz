import { expect, test, type Page } from "@playwright/test";

async function replaceInputs(page: Page, inputs: string[]) {
  const removeButtons = page.getByTestId("remove-input");
  await expect(page.getByTestId("input-row")).toHaveCount(12);

  for (let count = await removeButtons.count(); count > 0; count -= 1) {
    await removeButtons.first().click();
    await expect(removeButtons).toHaveCount(count - 1);
  }

  const composer = page.getByTestId("input-composer");
  for (const input of inputs) {
    await composer.fill(input);
    await composer.press("Enter");
  }
}

test("enter snippets, run PCA, select a point", async ({ page }) => {
  await page.goto("/?mockEmbeddings=1");

  await replaceInputs(page, ["alpha forest", "beta weather", "gamma cliffs"]);

  await page.getByTestId("reduction-PCA").click();
  await page.getByTestId("run-projection").click();

  await expect(page.getByText("paraphrase-MiniLM-L3-v2 · 3 points")).toBeVisible();
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

test("selecting CLIP keeps text inputs runnable", async ({ page }) => {
  await page.goto("/?mockEmbeddings=1");

  await page.getByLabel("Model").selectOption("Xenova/clip-vit-base-patch32");

  await expect(page.getByText("CLIP text/image encoders · ONNX ready")).toBeVisible();
  await expect(page.getByText("12 items")).toBeVisible();
  await expect(page.getByTestId("token-plan-overview")).toContainText("Images route direct");
  await expect(page.getByTestId("input-composer")).toBeEditable();
  await expect(page.getByTestId("run-projection")).toBeEnabled();
  await expect(page.getByLabel("Output")).toHaveValue("final");
});

test("dropped image inputs reveal a hover preview", async ({ page }) => {
  await page.goto("/?mockEmbeddings=1");

  await page.getByLabel("Model").selectOption("Xenova/clip-vit-base-patch32");
  await page.getByTestId("input-composer").dispatchEvent("drop", {
    dataTransfer: await page.evaluateHandle(() => {
      const transfer = new DataTransfer();
      const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="12"><rect width="16" height="12" fill="#2563eb"/></svg>`;
      transfer.items.add(new File([svg], "hover-preview.svg", { type: "image/svg+xml" }));
      return transfer;
    }),
  });

  const imageRow = page.getByTestId("input-row").filter({ hasText: "hover-preview.svg" });
  await expect(imageRow).toBeVisible();
  await expect(imageRow.getByTestId("image-hover-preview")).toBeHidden();

  await imageRow.hover();

  await expect(imageRow.getByTestId("image-hover-preview")).toBeVisible();
  const previewImage = imageRow.locator(".imageHoverPreview img");
  await expect(previewImage).toHaveAttribute("src", /^blob:/);
  await expect(previewImage).toHaveJSProperty("naturalWidth", 16);
});

test("image inputs stay listed but disabled after switching to a text-only model", async ({ page }) => {
  await page.goto("/?mockEmbeddings=1");

  await page.getByLabel("Model").selectOption("Xenova/clip-vit-base-patch32");
  await page.getByTestId("input-composer").dispatchEvent("drop", {
    dataTransfer: await page.evaluateHandle(() => {
      const transfer = new DataTransfer();
      const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="12"><rect width="16" height="12" fill="#2563eb"/></svg>`;
      transfer.items.add(new File([svg], "kept-image.svg", { type: "image/svg+xml" }));
      return transfer;
    }),
  });

  const imageRow = page.getByTestId("input-row").filter({ hasText: "kept-image.svg" });
  await expect(imageRow).toBeVisible();
  await expect(page.getByText("13 items")).toBeVisible();

  await page.getByLabel("Model").selectOption("Xenova/paraphrase-MiniLM-L3-v2");

  await expect(imageRow).toBeVisible();
  await expect(imageRow).toHaveAttribute("aria-disabled", "true");
  await expect(imageRow.getByTestId("input-item-meta")).toContainText("Unsupported");
  await expect(imageRow.getByTestId("input-item-meta")).toContainText("Not embedded by paraphrase-MiniLM-L3-v2");
  await expect(page.getByText("12 items")).toBeVisible();

  await page.getByTestId("run-projection").click();

  await expect(page.getByText("paraphrase-MiniLM-L3-v2 · 12 points")).toBeVisible();
  await expect(page.getByTestId("selected-point-label")).not.toHaveText("kept-image.svg");
});

test("dropped PDFs are accepted as chunked text inputs", async ({ page }) => {
  await page.goto("/?mockEmbeddings=1");

  await page.getByTestId("input-composer").dispatchEvent("drop", {
    dataTransfer: await page.evaluateHandle(() => {
      const transfer = new DataTransfer();
      transfer.items.add(new File(["%PDF-1.4 searchable text"], "research-notes.pdf", { type: "application/pdf" }));
      return transfer;
    }),
  });

  const pdfRow = page.getByTestId("input-row").filter({ hasText: "research-notes.pdf" });
  await expect(page.getByText("13 items")).toBeVisible();
  await expect(pdfRow).toContainText("PDF · application/pdf");

  await page.getByTestId("run-projection").click();

  await expect(pdfRow.getByTestId("input-item-meta")).toContainText("1 chunk");
});

test("files can be dropped anywhere in the app", async ({ page }) => {
  await page.goto("/?mockEmbeddings=1");

  const dataTransfer = await page.evaluateHandle(() => {
    const transfer = new DataTransfer();
    transfer.items.add(new File(["global drop text"], "global-notes.md", { type: "text/markdown" }));
    return transfer;
  });

  await page.locator(".rightPanel").dispatchEvent("dragenter", { dataTransfer });
  await expect(page.getByTestId("global-drop-overlay")).toContainText("Drop files to add them");

  await page.locator(".rightPanel").dispatchEvent("drop", { dataTransfer });

  await expect(page.getByTestId("global-drop-overlay")).toHaveCount(0);
  await expect(page.getByTestId("input-row").filter({ hasText: "global-notes.md" })).toBeVisible();
  await expect(page.getByText("13 items")).toBeVisible();
});

test("unsupported file drags show the unsupported overlay", async ({ page }) => {
  await page.goto("/?mockEmbeddings=1");

  const dataTransfer = await page.evaluateHandle(() => {
    const transfer = new DataTransfer();
    transfer.items.add(new File(["fake image"], "unsupported-image.png", { type: "image/png" }));
    return transfer;
  });

  await page.locator(".rightPanel").dispatchEvent("dragenter", { dataTransfer });

  await expect(page.getByTestId("global-drop-overlay")).toContainText("File is not supported");
  await expect(page.getByTestId("global-drop-overlay")).toContainText("Supported files: text and PDF.");
});

test("selected marker stays aligned with the canvas point in a stretched plot", async ({ page }) => {
  await page.setViewportSize({ width: 1920, height: 1100 });
  await page.goto("/?mockEmbeddings=1");

  await replaceInputs(page, ["alpha forest", "beta weather", "gamma cliffs"]);

  await page.getByTestId("reduction-PCA").click();
  await page.getByTestId("run-projection").click();
  await expect(page.getByTestId("selected-point-label")).toHaveText("alpha forest");

  const canvasBox = await page.getByTestId("point-cloud-canvas").boundingBox();
  const markerBox = await page.getByTestId("selected-point-marker").boundingBox();
  expect(canvasBox).not.toBeNull();
  expect(markerBox).not.toBeNull();

  const expectedX = canvasBox!.x + (canvasBox!.width * 120) / 860;
  const expectedY = canvasBox!.y + (canvasBox!.height * 320) / 640;
  const markerCenterX = markerBox!.x + markerBox!.width / 2;
  const markerCenterY = markerBox!.y + markerBox!.height / 2;

  expect(Math.abs(markerCenterX - expectedX)).toBeLessThan(2);
  expect(Math.abs(markerCenterY - expectedY)).toBeLessThan(2);
});

test("axis tick labels update after zooming the projection", async ({ page }) => {
  await page.goto("/?mockEmbeddings=1");

  await replaceInputs(page, ["alpha forest", "beta weather", "gamma cliffs"]);

  await page.getByTestId("run-projection").click();
  await expect(page.getByTestId("selected-point-label")).toHaveText("alpha forest");

  await expect(page.getByTestId("x-axis-tick")).toHaveText(["-6", "-4", "-2", "0", "2", "4", "6"]);

  await page.getByTitle("Zoom in").click();

  await expect(page.getByTestId("x-axis-tick")).toHaveText(["-4", "-2", "0", "2", "4"]);
});

test("selected marker is only shown when the selected point is rendered", async ({ page }) => {
  await page.goto("/?mockEmbeddings=1");

  await replaceInputs(page, ["alpha forest", "beta weather", "gamma cliffs"]);

  await page.getByTestId("run-projection").click();
  await expect(page.getByTestId("selected-point-label")).toHaveText("alpha forest");
  await expect(page.getByTestId("selected-point-marker")).toHaveCount(1);

  await page.getByLabel("Search points").fill("beta");

  await expect(page.getByTestId("selected-point-label")).toHaveText("alpha forest");
  await expect(page.getByTestId("selected-point-marker")).toHaveCount(0);
});

test("run includes text still sitting in the composer", async ({ page }) => {
  await page.goto("/?mockEmbeddings=1");

  await page.getByTestId("input-composer").fill("delta draft input");

  await expect(page.getByText("13 items")).toBeVisible();
  await page.getByTestId("run-projection").click();

  await expect(page.getByText("paraphrase-MiniLM-L3-v2 · 13 points")).toBeVisible();
  await expect(page.getByTestId("selected-point-label")).toHaveText("Thunderstorms rolled across the harbor before sunrise.");
  await expect(page.getByTestId("input-composer")).toHaveValue("");
  await expect(page.getByTestId("input-row").filter({ hasText: "delta draft input" }).locator("strong")).toHaveText("delta draft input");
});

test("hiding the only run clears hidden selected point details", async ({ page }) => {
  await page.goto("/?mockEmbeddings=1");

  await replaceInputs(page, ["alpha forest", "beta weather", "gamma cliffs"]);
  await page.getByTestId("run-projection").click();
  await expect(page.getByTestId("selected-point-label")).toHaveText("alpha forest");

  await page.locator(".runCard input[type='checkbox']").uncheck();

  await expect(page.getByText("No visible points. Turn a run back on to show embeddings.")).toBeVisible();
  await expect(page.getByText("No point selected.")).toBeVisible();
  await expect(page.getByTestId("selected-point-label")).toHaveCount(0);
});

test("narrow viewports do not force desktop horizontal scrolling", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/?mockEmbeddings=1");

  const metrics = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    viewportWidth: window.innerWidth,
  }));

  expect(metrics.scrollWidth).toBeLessThanOrEqual(metrics.viewportWidth + 2);
});

test("input widget shows token plan totals and per-snippet metadata", async ({ page }) => {
  await page.goto("/?mockEmbeddings=1");

  await replaceInputs(page, ["alpha forest", "beta weather", "gamma cliffs"]);

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

  await replaceInputs(page, [longLabel, "beta weather", "gamma cliffs"]);

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

test("ambient labels stay suppressed and search labels stay inside the visible chart area", async ({ page }) => {
  await page.goto("/?mockEmbeddings=1");

  await replaceInputs(page, [
    "left edge label with enough text to exercise truncation",
    "center label",
    "right edge label with enough text to exercise truncation",
  ]);

  await page.getByTestId("run-projection").click();
  await expect(page.getByTestId("selected-point-label")).toHaveText(
    "left edge label with enough text to exercise truncation",
  );
  await expect(page.getByTestId("point-label")).toHaveCount(0);
  await page.getByLabel("Search points").fill("label");

  const plotBox = await page.locator(".plotCanvas").boundingBox();
  expect(plotBox).not.toBeNull();

  const labelRects = await page.getByTestId("point-label").evaluateAll((labels) =>
    labels.map((label) => {
      const rect = label.getBoundingClientRect();
      return {
        left: rect.left,
        right: rect.right,
        top: rect.top,
        bottom: rect.bottom,
        text: Array.from(label.childNodes)
          .filter((node) => node.nodeType === Node.TEXT_NODE)
          .map((node) => node.textContent)
          .join("")
          .trim(),
      };
    }),
  );

  expect(labelRects.length).toBeGreaterThan(0);
  for (const rect of labelRects) {
    expect(rect.left).toBeGreaterThanOrEqual(plotBox!.x - 1);
    expect(rect.right).toBeLessThanOrEqual(plotBox!.x + plotBox!.width + 1);
    expect(rect.top).toBeGreaterThanOrEqual(plotBox!.y - 1);
    expect(rect.bottom).toBeLessThanOrEqual(plotBox!.y + plotBox!.height + 1);
    expect(rect.text?.length ?? 0).toBeLessThanOrEqual(32);
  }
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

  await replaceInputs(page, ["alpha forest", "beta weather", "gamma cliffs"]);

  await page.getByTestId("run-projection").click();
  await expect(page.getByTestId("selected-point-label")).toHaveText("previous close");
  await page.getByTestId("run-projection").click();
  await expect(page.getByTestId("selected-point-label")).toHaveText("current anchor");

  const nearestLabels = page.getByTestId("nearest-row").locator("strong");
  await expect(nearestLabels.nth(0)).toHaveText("previous close");
  await expect(nearestLabels.nth(1)).toHaveText("current mid");
  await expect(nearestLabels.nth(2)).toHaveText("previous mid");
});

test("wheel zoom only captures scrolling while the left mouse button is down", async ({ page }) => {
  await page.goto("/?mockEmbeddings=1");

  await replaceInputs(page, ["alpha forest", "beta weather", "gamma cliffs"]);
  await page.getByTestId("run-projection").click();
  await expect(page.getByTestId("selected-point-label")).toHaveText("alpha forest");

  const canvas = page.getByTestId("point-cloud-canvas");
  const box = await canvas.boundingBox();
  expect(box).not.toBeNull();

  const selectedMarker = page.getByTestId("selected-point-marker");
  const initialX = await selectedMarker.getAttribute("cx");

  await canvas.hover({
    position: {
      x: box!.width / 2,
      y: box!.height / 2,
    },
  });
  await page.mouse.wheel(0, -200);
  await expect(selectedMarker).toHaveAttribute("cx", initialX!);

  await page.mouse.move(box!.x + box!.width / 2, box!.y + box!.height / 2);
  await page.mouse.down({ button: "left" });
  await page.mouse.wheel(0, -200);
  await page.mouse.up({ button: "left" });

  await expect(selectedMarker).not.toHaveAttribute("cx", initialX!);
});
