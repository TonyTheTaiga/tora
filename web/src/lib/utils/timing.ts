export function startTimer(action?: string, metadata?: any) {
  const start = performance.now();
  return {
    stop: () => performance.now() - start,
    elapsed: () => performance.now() - start,
    end: () => performance.now() - start
  };
}