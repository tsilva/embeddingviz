import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const packageJson = JSON.parse(await readFile(new URL("../package.json", import.meta.url), "utf8"));
const packageLock = JSON.parse(await readFile(new URL("../package-lock.json", import.meta.url), "utf8"));

function numericVersion(version) {
  return version.split(/[.-]/).slice(0, 3).map(Number);
}

function versionAtLeast(version, floor) {
  const current = numericVersion(version);
  const required = numericVersion(floor);
  for (let index = 0; index < 3; index += 1) {
    const difference = (current[index] ?? 0) - (required[index] ?? 0);
    if (difference !== 0) return difference > 0;
  }
  return true;
}

const floors = new Map([
  ["@babel/core", "7.29.6"],
  ["adm-zip", "0.6.1"],
  ["nanoid", "3.3.18"],
  ["pdfjs-dist", "6.2.108"],
  ["postcss", "8.5.23"],
  ["protobufjs", "7.6.5"],
  ["sharp", "0.35.5"],
  ["undici", "7.29.1"],
  ["vite", "8.0.16"],
]);

for (const [path, dependency] of Object.entries(packageLock.packages)) {
  if (!path) continue;
  assert.match(dependency.resolved ?? "", /^https:\/\/registry\.npmjs\.org\//);
  const name = path.replace(/^node_modules\//, "").replace(/^.*\/node_modules\//, "");
  if (name === "brace-expansion") {
    const major = numericVersion(dependency.version)[0];
    assert.ok(major === 1 || major === 5, `unexpected brace-expansion major ${dependency.version}`);
    const floor = major === 1 ? "1.1.18" : "5.0.9";
    assert.ok(versionAtLeast(dependency.version, floor), `${name}@${dependency.version} is below ${floor}`);
  }
  const floor = floors.get(name);
  if (floor) {
    assert.ok(versionAtLeast(dependency.version, floor), `${name}@${dependency.version} is below ${floor}`);
  }
}

for (const dependencies of [packageJson.dependencies, packageJson.devDependencies]) {
  for (const specifier of Object.values(dependencies ?? {})) {
    assert.doesNotMatch(specifier, /^(?:git(?:\+|:)|https?:|file:|link:|workspace:)/i);
  }
}
assert.deepEqual(packageJson.overrides, {
  "source-map-js": "1.2.2",
  "@huggingface/transformers": {
    "onnxruntime-node": { "adm-zip": "0.6.1" },
    "sharp": "0.35.5",
  },
});

const transformersRequire = createRequire(require.resolve("@huggingface/transformers"));
const sharp = transformersRequire("sharp");
const onnxRequire = createRequire(transformersRequire.resolve("onnxruntime-node"));
const AdmZip = onnxRequire("adm-zip");

const png = await sharp({
  create: { width: 2, height: 2, channels: 3, background: { r: 10, g: 20, b: 30 } },
}).png().toBuffer();
assert.deepEqual(await sharp(png).metadata().then(({ width, height, format }) => ({ width, height, format })), {
  width: 2,
  height: 2,
  format: "png",
});
await assert.rejects(() => sharp(Buffer.from("not an image")).metadata());

const archive = new AdmZip();
archive.addFile("model.txt", Buffer.from("safe model fixture"));
const reopened = new AdmZip(archive.toBuffer());
assert.equal(reopened.readAsText("model.txt"), "safe model fixture");

const transformers = await import("@huggingface/transformers");
assert.equal(typeof transformers.pipeline, "function");
