<script lang="ts">
  import { getToasts, removeToast, type Toast } from "$lib/state/toast.svelte";
  import {
    X,
    CheckCircle,
    AlertCircle,
    Info,
    AlertTriangle,
  } from "@lucide/svelte";

  let toasts = $derived(getToasts());

  const icons = {
    success: CheckCircle,
    error: AlertCircle,
    info: Info,
    warning: AlertTriangle,
  };

  const colors = {
    success: "border-ctp-green bg-ctp-green/10 text-ctp-green",
    error: "border-ctp-red bg-ctp-red/10 text-ctp-red",
    info: "border-ctp-blue bg-ctp-blue/10 text-ctp-blue",
    warning: "border-ctp-yellow bg-ctp-yellow/10 text-ctp-yellow",
  };
</script>

<div
  class="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-sm"
  role="region"
  aria-label="Notifications"
>
  {#each toasts as toast (toast.id)}
    {@const Icon = icons[toast.type]}
    <div
      class="flex items-start gap-3 p-3 border backdrop-blur-sm shadow-lg {colors[
        toast.type
      ]} layer-slide-up"
      role="alert"
      aria-live="polite"
    >
      <Icon size={18} class="flex-shrink-0 mt-0.5" />
      <p class="flex-1 text-sm">{toast.message}</p>
      <button
        onclick={() => removeToast(toast.id)}
        class="text-current opacity-60 hover:opacity-100 transition-opacity"
        aria-label="Dismiss notification"
      >
        <X size={16} />
      </button>
    </div>
  {/each}
</div>
