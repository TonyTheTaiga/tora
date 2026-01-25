export type ToastType = "success" | "error" | "info" | "warning";

export interface Toast {
  id: string;
  type: ToastType;
  message: string;
  duration: number;
}

let toasts = $state<Toast[]>([]);
let counter = 0;

export function getToasts(): Toast[] {
  return toasts;
}

export function addToast(
  type: ToastType,
  message: string,
  duration = 4000,
): string {
  const id = `toast-${++counter}`;
  const toast: Toast = { id, type, message, duration };

  toasts = [...toasts, toast];

  if (duration > 0) {
    setTimeout(() => removeToast(id), duration);
  }

  return id;
}

export function removeToast(id: string): void {
  toasts = toasts.filter((t) => t.id !== id);
}

// Convenience functions
export const toast = {
  success: (msg: string, duration?: number) =>
    addToast("success", msg, duration),
  error: (msg: string, duration?: number) => addToast("error", msg, duration),
  info: (msg: string, duration?: number) => addToast("info", msg, duration),
  warning: (msg: string, duration?: number) =>
    addToast("warning", msg, duration),
};
