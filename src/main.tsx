import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Analytics } from "@vercel/analytics/react";
import App from "./App";
import { Sentry } from "./sentry";
import "./styles.css";

createRoot(document.getElementById("root")!, {
  onCaughtError: Sentry.reactErrorHandler(),
  onRecoverableError: Sentry.reactErrorHandler(),
  onUncaughtError: Sentry.reactErrorHandler(),
}).render(
  <StrictMode>
    <App />
    <Analytics />
  </StrictMode>,
);
