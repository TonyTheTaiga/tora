export function generateRequestId(): string {
  return (
    Math.random().toString(36).substring(2, 15) +
    Math.random().toString(36).substring(2, 15)
  );
}

type TimingMetadata = Record<string, unknown> | undefined;

export function startTimer(action?: string, metadata?: TimingMetadata) {
  const start = performance.now();
  const label = action || "timer";
  const baseMeta = metadata || {};

  function nowMs() {
    return performance.now();
  }

  function safeLog(kind: "info" | "error", message: string, meta?: any) {
    try {
      if (kind === "info") {
        if (meta && Object.keys(meta).length) {
          console.info(message, meta);
        } else {
          console.info(message);
        }
      } else {
        if (meta && Object.keys(meta).length) {
          console.error(message, meta);
        } else {
          console.error(message);
        }
      }
    } catch {
      // never throw from logging
    }
  }

  return {
    stop: () => nowMs() - start,
    elapsed: () => nowMs() - start,
    end: (endMetadata?: TimingMetadata) => {
      const elapsed = nowMs() - start;
      const meta = { ...baseMeta, ...(endMetadata || {}) } as Record<
        string,
        unknown
      >;
      const msg = `[timing] ${label} in ${elapsed.toFixed(1)}ms`;
      safeLog("info", msg, meta);
      return elapsed;
    },
  };
}
