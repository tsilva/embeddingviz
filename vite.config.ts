/// <reference types="vitest/config" />
import { sentryVitePlugin } from "@sentry/vite-plugin";
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const hasSentrySourceMapConfig = Boolean(
    env.SENTRY_AUTH_TOKEN && env.SENTRY_ORG && env.SENTRY_PROJECT,
  );
  const enableSentrySourceMaps = Boolean(
    hasSentrySourceMapConfig &&
      (env.SENTRY_UPLOAD_SOURCEMAPS === "true" || env.CI === "true"),
  );

  return {
    build: {
      sourcemap: enableSentrySourceMaps,
    },
    plugins: [
      react(),
      enableSentrySourceMaps &&
        sentryVitePlugin({
          authToken: env.SENTRY_AUTH_TOKEN,
          org: env.SENTRY_ORG,
          project: env.SENTRY_PROJECT,
          url: env.SENTRY_BASE_URL || "https://sentry.io",
          sourcemaps: {
            filesToDeleteAfterUpload: ["dist/**/*.js.map"],
          },
        }),
    ],
    optimizeDeps: {
      exclude: ["@huggingface/transformers"],
    },
    test: {
      environment: "jsdom",
      globals: true,
      include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
    },
  };
});
