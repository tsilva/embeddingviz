import { StrictMode } from "react";
import { createRoot, hydrateRoot } from "react-dom/client";
import { Analytics } from "@vercel/analytics/react";
import App from "./App";
import { Sentry } from "./sentry";
import "./styles.css";

const root = document.getElementById("root")!;
const options = {
  onCaughtError: Sentry.reactErrorHandler(),
  onRecoverableError: Sentry.reactErrorHandler(),
  onUncaughtError: Sentry.reactErrorHandler(),
};
const app = (
  <StrictMode>
    <App />
    <Analytics />
  </StrictMode>
);

if (root.hasChildNodes()) {
  hydrateRoot(root, app, options);
} else {
  createRoot(root, options).render(app);
}
