import { readFile, writeFile } from "node:fs/promises";
import { createElement } from "react";
import { renderToString } from "react-dom/server";
import react from "@vitejs/plugin-react";
import { createServer } from "vite";

const renderer = await createServer({
  configFile: false,
  mode: "production",
  appType: "custom",
  plugins: [react()],
  server: { middlewareMode: true, ws: false, watch: null },
});

try {
  const { default: App } = await renderer.ssrLoadModule("/src/App.tsx");
  const output = new URL("../dist/index.html", import.meta.url);
  const html = await readFile(output, "utf8");
  const placeholder = '<div id="root"></div>';
  if (!html.includes(placeholder)) throw new Error("Missing EmbeddingViz prerender root");

  const markup = renderToString(createElement(App));
  await writeFile(output, html.replace(placeholder, () => `<div id="root">${markup}</div>`));
} finally {
  await renderer.close();
}
