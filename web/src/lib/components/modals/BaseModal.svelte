<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { X } from "@lucide/svelte";

  let {
    title,
    children,
    onClose,
    size = "lg",
  }: {
    title: string;
    children: any;
    onClose?: () => void;
    size?: "sm" | "md" | "lg";
  } = $props();

  onMount(() => {
    document.body.classList.add("overflow-hidden");
  });

  onDestroy(() => {
    document.body.classList.remove("overflow-hidden");
  });
</script>

<div
  class="fixed inset-0 bg-ctp-mantle/90 backdrop-blur-sm
         flex items-center justify-center p-4 z-50 overflow-hidden font-mono"
>
  <div
    class={`w-full ${
      size === "sm" ? "max-w-sm" : size === "md" ? "max-w-md" : "max-w-xl"
    } bg-ctp-mantle border border-ctp-surface0/30 overflow-auto overflow-x-hidden max-h-[90vh]`}
    role="dialog"
    aria-modal="true"
    aria-labelledby="modal-title"
  >
    <div
      class="flex items-center justify-between p-6 border-b border-ctp-surface0/20"
    >
      <div class="flex items-center gap-3">
        <div class="w-1 h-5 bg-ctp-blue"></div>
        <h3 id="modal-title" class="text-lg font-bold text-ctp-text">
          {title}
        </h3>
      </div>
      {#if onClose}
        <button
          onclick={onClose}
          type="button"
          class="text-ctp-subtext0 hover:text-ctp-text p-1 transition-colors"
          aria-label="Close modal"
        >
          <X size={20} />
        </button>
      {/if}
    </div>

    <div class="px-6 py-4">
      {@render children()}
    </div>
  </div>
</div>
