interface TimingContext {
  requestId?: string;
  userId?: string;
  experimentId?: string;
  [key: string]: string | undefined;
}

class Timer {
  private startTime: number;
  private operation: string;
  private context: TimingContext;

  constructor(operation: string, context: TimingContext = {}) {
    this.operation = operation;
    this.context = context;
    this.startTime = performance.now();
  }

  end(additionalContext?: Record<string, string | number>): number {
    const duration = performance.now() - this.startTime;
    
    const logData = {
      operation: this.operation,
      duration: Math.round(duration * 100) / 100,
      ...this.context,
      ...additionalContext,
    };

    if (typeof window !== 'undefined') {
      console.log(`[TIMING:CLIENT] ${this.operation}:`, logData);
    } else {
      console.log(`[TIMING:SERVER] ${this.operation}:`, JSON.stringify(logData));
    }

    return duration;
  }
}

export function startTimer(operation: string, context: TimingContext = {}): Timer {
  return new Timer(operation, context);
}

export function timeAsync<T>(
  operation: string,
  fn: () => Promise<T>,
  context: TimingContext = {}
): Promise<T> {
  const timer = startTimer(operation, context);
  return fn().finally(() => timer.end());
}

export function timeSync<T>(
  operation: string,
  fn: () => T,
  context: TimingContext = {}
): T {
  const timer = startTimer(operation, context);
  try {
    return fn();
  } finally {
    timer.end();
  }
}

export function generateRequestId(): string {
  return Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
}