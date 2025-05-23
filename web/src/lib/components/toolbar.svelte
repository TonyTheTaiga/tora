<script lang="ts">
  import { Plus, User } from "lucide-svelte";
  import ThemeToggle from "./theme-toggle.svelte";
  import { goto } from "$app/navigation";
  import { page } from "$app/state";
  import type { Attachment } from "svelte/attachments";
  import tippy from "tippy.js";

  let {
    selectedExperiment = $bindable(),
    isOpenCreate = $bindable(),
    hasExperiments = $bindable(),
  } = $props();
  let { session } = $derived(page.data);

  function tooltip(content: string, check: () => boolean): Attachment {
    return (element) => {
      if (check()) {
        const tippyInstance = tippy(element, {
          content: `${content}
                    <div style="text-align: center;"><svg class="arrow-down" width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M12 16l-6-6h12l-6 6z"/></svg></div>`,
          allowHTML: true,
          trigger: "manual",
          hideOnClick: false,
          showOnCreate: true,
          theme: "tooltip-theme",
          animation: "scale",
          duration: [200, 150],
          placement: "top",
          arrow: true,
          offset: [0, 10],
          maxWidth: 200,
          zIndex: 35,
        });
        return tippyInstance.destroy;
      }
    };
  }
</script>

<div
  class="
    fixed
    bottom-12
    left-1/2
    transform -translate-x-1/2
    flex flex-row items-center gap-1
    bg-ctp-surface1 border border-ctp-surface2
    rounded-lg shadow-md z-40
    overflow-hidden
    sm:scale-100 sm:opacity-100
    md:scale-120 md:hover:scale-140 md:transition-transform md:duration-300
    md:opacity-80 md:hover:opacity-100
    lg:scale-140 lg:hover:scale-160 lg:transition-transform lg:duration-300
    lg:opacity-80 lg:hover:opacity-100
  "
>
  <button
    class="p-1.5 text-ctp-subtext0 hover:text-ctp-text hover:bg-ctp-surface2 transition-colors"
    title="Create a new experiment"
    onclick={() => {
      isOpenCreate = true;
    }}
    {@attach tooltip(
      "<div>No experiments yet! <strong>Ready to start? Tap here!</strong> and kick things off!</div>",
      () => {
        return hasExperiments === false;
      },
    )}
  >
    <Plus size={16} />
  </button>

  <ThemeToggle />

  {#if session && session.user}
    <button
      class="p-1.5 text-ctp-subtext0 hover:text-ctp-text hover:bg-ctp-surface2 transition-colors"
      title="Go to user profile"
      onclick={() => {
        goto(`/users/${session.user.id}`);
      }}
    >
      <User size={16} />
    </button>
  {/if}
</div>

<style>
  :global(.tippy-box[data-theme~="tooltip-theme"]) {
    background-color: var(--color-ctp-surface1);
    border: 1px solid var(--color-ctp-surface2);
    border-radius: 0.5rem;
    box-shadow:
      0 4px 6px -1px rgba(0, 0, 0, 0.1),
      0 2px 4px -1px rgba(0, 0, 0, 0.06);
    backdrop-filter: blur(8px);
    animation:
      tooltipFloat 3s ease-in-out infinite,
      tooltipGlow 3s ease-in-out infinite;
  }

  :global(.tippy-box[data-theme~="tooltip-theme"] .tippy-content) {
    color: var(--color-ctp-text);
    font-size: 0.875rem;
    line-height: 1.4;
    padding: 0.75rem 1rem;
    font-weight: 500;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
  }

  :global(.tippy-box[data-theme~="tooltip-theme"] .arrow-down) {
    color: var(--color-ctp-blue);
    animation: arrowPulse 1.5s ease-in-out infinite;
  }

  :global(.tippy-box[data-theme~="tooltip-theme"] .tippy-arrow) {
    color: var(
      --color-ctp-surface1
    ); /* Matches toolbar background for the arrow */
  }

  :global(.tippy-box[data-theme~="tooltip-theme"] .tippy-arrow::before) {
    border-color: var(
      --color-ctp-surface2
    ); /* Matches toolbar border for the arrow */
  }

  :global(.tippy-box[data-theme~="tooltip-theme"]::before) {
    content: "";
    position: absolute;
    top: -1px;
    left: -1px;
    right: -1px;
    bottom: -1px;
    background: var(--color-ctp-blue);
    border-radius: 0.625rem;
    z-index: -1;
    opacity: 0.2;
  }

  @keyframes tooltipFloat {
    0%,
    100% {
      transform: translateY(0px);
    }
    50% {
      transform: translateY(-5px);
    }
  }

  @keyframes tooltipGlow {
    0%,
    100% {
      box-shadow:
        0 4px 6px -1px rgba(0, 0, 0, 0.1),
        0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    50% {
      box-shadow:
        0 10px 15px -3px rgba(0, 0, 0, 0.1),
        0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
  }

  @keyframes arrowPulse {
    0%,
    100% {
      transform: translateY(0px);
      opacity: 0.7;
    }
    50% {
      transform: translateY(3px);
      opacity: 1;
    }
  }
</style>
