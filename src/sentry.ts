import * as Sentry from "@sentry/react";

const sentryDsn = import.meta.env.VITE_SENTRY_DSN;

const parseSampleRate = (value: string | undefined, fallback: number) => {
  if (!value) {
    return fallback;
  }

  const sampleRate = Number(value);

  if (!Number.isFinite(sampleRate)) {
    return fallback;
  }

  return Math.min(Math.max(sampleRate, 0), 1);
};

if (sentryDsn) {
  Sentry.init({
    dsn: sentryDsn,
    environment:
      import.meta.env.VITE_SENTRY_ENVIRONMENT || import.meta.env.MODE,
    release: import.meta.env.VITE_SENTRY_RELEASE || undefined,
    integrations: [Sentry.browserTracingIntegration()],
    tracesSampleRate: parseSampleRate(
      import.meta.env.VITE_SENTRY_TRACES_SAMPLE_RATE,
      import.meta.env.PROD ? 0.1 : 1.0,
    ),
    tracePropagationTargets: [/^\//],
  });
}

export { Sentry };
