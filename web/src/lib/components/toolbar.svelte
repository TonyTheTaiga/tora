<script lang="ts">
  import { Plus, User, Briefcase, Moon, Sun } from "lucide-svelte";
  import { goto } from "$app/navigation";
  import { page } from "$app/state";
  import type { Attachment } from "svelte/attachments";
  import tippy from "tippy.js";
  import { onMount, onDestroy } from "svelte";

  let {
    selectedExperiment = $bindable(),
    isOpenCreate = $bindable(),
    hasExperiments = $bindable(),
  } = $props();
  let { session } = $derived(page.data);
  let theme = $state<"dark" | "light">("dark");
  let isAtBottom = $state(false);

  const handleScroll = () => {
    if (typeof window !== 'undefined' && typeof document !== 'undefined') {
      const pageIsScrollable = document.documentElement.scrollHeight > window.innerHeight;
      const atActualBottom = (window.innerHeight + Math.ceil(window.scrollY)) >= document.documentElement.scrollHeight - 1;

      if (pageIsScrollable && atActualBottom) {
        isAtBottom = true;
      } else {
        isAtBottom = false;
      }
    }
  };

  onMount(() => {
    if (typeof window !== 'undefined') {
      window.addEventListener('scroll', handleScroll);
      window.addEventListener('resize', handleScroll);
      handleScroll();
    }
    if (typeof window !== 'undefined' && typeof localStorage !== 'undefined' && typeof document !== 'undefined') {
      const savedTheme = localStorage.getItem("theme");
      if (savedTheme && (savedTheme === "dark" || savedTheme === "light")) {
        theme = savedTheme as "dark" | "light";
        applyTheme(theme);
      } else {
        const prefersDark = window.matchMedia(
          "(prefers-color-scheme: dark)",
        ).matches;
        theme = prefersDark ? "dark" : "light";
        applyTheme(theme);
      }
    }
  });

  onDestroy(() => {
    if (typeof window !== 'undefined') {
      window.removeEventListener('scroll', handleScroll);
      window.removeEventListener('resize', handleScroll); 
    }
  });

  function toggleTheme() {
    theme = theme === "dark" ? "light" : "dark";
    if (typeof window !== 'undefined' && typeof localStorage !== 'undefined' && typeof document !== 'undefined') {
      applyTheme(theme);
      localStorage.setItem("theme", theme);
    }
  }

  function applyTheme(newTheme: "dark" | "light") {
    if (typeof document !== 'undefined') {
      if (newTheme === "dark") {
        document.documentElement.classList.remove("light");
        document.documentElement.classList.add("dark");
      } else {
        document.documentElement.classList.remove("dark");
        document.documentElement.classList.add("light");
      }
    }
  }

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

<div class="toolbar" class:hidden-at-bottom={isAtBottom}>
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

  <button
    onclick={() => { toggleTheme(); }}
    class="toolbar-button"
    aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
    title={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
  >
    {#if theme === "dark"}
      <Sun class="icon" />
    {:else}
      <Moon class="icon" />
    {/if}
  </button>

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
    );
  }

  :global(.tippy-box[data-theme~="tooltip-theme"] .tippy-arrow::before) {
    border-color: var(
      --color-ctp-surface2
    );  
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

  .toolbar {
    position: fixed;
    bottom: 1rem;
    left: 50vw;
    transform: translateX(-50%) scale(1.25);
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
    opacity: 100%;
    transition: transform 0.3s ease, opacity 0.3s ease;
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
      transform: translateX(-50%) scale(1.1);
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
      width: auto;
      max-width: calc(100vw - 4rem);
      transform: translateX(-50%) scale(1.1);
      opacity: 80%;
    }
    
    .toolbar:hover {
      transform: translateX(-50%) scale(1.2);
      opacity: 100%;
    }
  }
  
  @media (min-width: 1024px) {
    .toolbar {
      transform: translateX(-50%) scale(1.2); 
      opacity: 80%;
    }
    
    .toolbar:hover {
      transform: translateX(-50%) scale(1.3);
      opacity: 100%;
    }
  }

  .toolbar.hidden-at-bottom {
    opacity: 0;
    transform: translateX(-50%) translateY(100%) scale(1.25); 
    pointer-events: none;
  }

  @media (min-width: 640px) {
    .toolbar.hidden-at-bottom {
      transform: translateX(-50%) translateY(100%) scale(1.1);
    }
  }

  @media (min-width: 768px) {
    .toolbar.hidden-at-bottom {
      transform: translateX(-50%) translateY(100%) scale(1.1);
    }
  }

  @media (min-width: 1024px) {
    .toolbar.hidden-at-bottom {
      transform: translateX(-50%) translateY(100%) scale(1.2);
    }
  }
</style>
