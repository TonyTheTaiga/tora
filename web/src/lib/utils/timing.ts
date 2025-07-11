export function generateRequestId(): string {
  return (
    Math.random().toString(36).substring(2, 15) +
    Math.random().toString(36).substring(2, 15)
  );
}

export function startTimer(_action?: string, _metadata?: any) {
  const start = performance.now();
  return {
    stop: () => performance.now() - start,
    elapsed: () => performance.now() - start,
    end: (_endMetadata?: any) => performance.now() - start,
  };
}
