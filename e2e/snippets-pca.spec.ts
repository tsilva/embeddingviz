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
