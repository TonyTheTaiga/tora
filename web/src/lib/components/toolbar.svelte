<script lang="ts">
  import { Plus, User, Briefcase } from "lucide-svelte";
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

<div class="toolbar">
  <button
    class="toolbar-button"
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
    <Plus class="icon" />
  </button>

  <ThemeToggle />

  {#if session && session.user}
    <button
      class="toolbar-button"
      title="Manage workspaces"
      onclick={() => {
        goto("/workspaces");
      }}
    >
      <Briefcase class="icon" />
    </button>

    <button
      class="toolbar-button"
      title="Go to user profile"
      onclick={() => {
        goto(`/users/${session.user.id}`);
      }}
    >
      <User class="icon" />
    </button>
  {/if}
</div>

<style>
  /* Existing tooltip styles */
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
    font-size: 0.75rem;
    line-height: 1.4;
    padding: 0.5rem 0.75rem;
    font-weight: 500;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.25rem;
  }

  @media (min-width: 640px) {
    :global(.tippy-box[data-theme~="tooltip-theme"] .tippy-content) {
      font-size: 0.875rem;
      padding: 0.75rem 1rem;
      gap: 0.5rem;
    }
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

  /* Toolbar styling */
  .toolbar {
    position: fixed;
    bottom: 1rem;
    left: 50vw;
    transform: translateX(-50%);
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 0.25rem;
    background-color: var(--color-ctp-surface1);
    border: 1px solid var(--color-ctp-surface2);
    border-radius: 0.5rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    z-index: 40;
    overflow: hidden;
    max-width: calc(100vw - 2rem);
    scale: 100%;
    opacity: 100%;
    transition: scale 0.3s ease, opacity 0.3s ease;
  }
  
  .toolbar-button {
    padding: 0.5rem;
    color: var(--color-ctp-subtext0);
    transition: color 0.2s, background-color 0.2s;
  }
  
  .toolbar-button:hover {
    color: var(--color-ctp-text);
    background-color: var(--color-ctp-surface2);
  }
  
  :global(.icon) {
    width: 20px;
    height: 20px;
  }
  
  /* Responsive styles */
  @media (min-width: 640px) {
    .toolbar {
      bottom: 3rem;
      gap: 0.25rem;
      scale: 110%;
      transition: scale 0.3s ease, opacity 0.3s ease;
    }
    
    .toolbar-button {
      padding: 0.625rem;
    }
    
    :global(.icon) {
      width: 20px;
      height: 20px;
    }
  }
  
  @media (min-width: 768px) {
    .toolbar {
      scale: 110%;
      opacity: 80%;
      transition: scale 0.3s ease, opacity 0.3s ease;
    }
    
    .toolbar:hover {
      scale: 120%;
      opacity: 100%;
    }
  }
  
  @media (min-width: 1024px) {
    .toolbar {
      scale: 120%;
      opacity: 80%;
      transition: scale 0.3s ease, opacity 0.3s ease;
    }
    
    .toolbar:hover {
      scale: 130%;
      opacity: 100%;
    }
  }
</style>
