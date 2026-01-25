<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { X } from "@lucide/svelte";

  const FOCUSABLE_SELECTORS =
    'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

  let {
    title,
    children,
    onClose,
    size = "lg",
    closeDisabled = false,
  }: {
    title: string;
    children: any;
    onClose?: () => void;
    size?: "sm" | "md" | "lg";
    closeDisabled?: boolean;
  } = $props();

  let modalContainer: HTMLDivElement | null = null;
  let previouslyFocused: HTMLElement | null = null;

  function handleKeyDown(e: KeyboardEvent) {
    // ESC to close
    if (e.key === "Escape" && onClose && !closeDisabled) {
      e.preventDefault();
      onClose();
      return;
    }

    // Tab focus trap
    if (e.key === "Tab" && modalContainer) {
      const focusable = Array.from(
        modalContainer.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTORS),
      );
      if (focusable.length === 0) return;

      const firstElement = focusable[0];
      const lastElement = focusable[focusable.length - 1];

      if (e.shiftKey && document.activeElement === firstElement) {
        e.preventDefault();
        lastElement.focus();
      } else if (!e.shiftKey && document.activeElement === lastElement) {
        e.preventDefault();
        firstElement.focus();
      }
    }
  }

  onMount(() => {
    document.body.classList.add("overflow-hidden");

    // Store previously focused element
    previouslyFocused = document.activeElement as HTMLElement;

    // Add keyboard listener
    document.addEventListener("keydown", handleKeyDown);

    // Auto-focus first focusable element
    requestAnimationFrame(() => {
      if (modalContainer) {
        const focusable =
          modalContainer.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTORS);
        if (focusable.length > 0) {
          focusable[0].focus();
        }
      }
    });
  });

  onDestroy(() => {
    document.body.classList.remove("overflow-hidden");
    document.removeEventListener("keydown", handleKeyDown);

    // Restore focus to previously focused element
    previouslyFocused?.focus();
  });
</script>

<div
  class="fixed inset-0 bg-ctp-mantle/90 backdrop-blur-sm
         flex items-center justify-center p-4 z-50 overflow-hidden"
>
  <div
    bind:this={modalContainer}
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
