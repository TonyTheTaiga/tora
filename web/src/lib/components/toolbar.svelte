<script lang="ts">
  import {
    Plus,
    User,
    Briefcase,
    Moon,
    Sun,
    GitCompareArrows,
    Cog,
  } from "lucide-svelte";
  import { goto } from "$app/navigation";
  import { page } from "$app/state";
  import { onMount, onDestroy } from "svelte";
  import { toggleMode } from "$lib/state/comparison.svelte.js";
  import {
    getTheme,
    toggleTheme as toggleAppTheme,
  } from "$lib/state/theme.svelte.js";
  import { openCreateExperimentModal } from "$lib/state/app.svelte.js";

  let { session } = $derived.by(() => page.data);
  let theme = $derived.by(() => getTheme());
  let isAtBottom = $state(false);

  const handleScroll = () => {
    if (typeof window !== "undefined" && typeof document !== "undefined") {
      const pageIsScrollable =
        document.documentElement.scrollHeight > window.innerHeight;
      const atActualBottom =
        window.innerHeight + Math.ceil(window.scrollY) >=
        document.documentElement.scrollHeight - 1;

      if (pageIsScrollable && atActualBottom) {
        isAtBottom = true;
      } else {
        isAtBottom = false;
      }
    }
  };

  onMount(() => {
    if (typeof window !== "undefined") {
      window.addEventListener("scroll", handleScroll);
      window.addEventListener("resize", handleScroll);
      handleScroll();
    }
  });

  onDestroy(() => {
    if (typeof window !== "undefined") {
      window.removeEventListener("scroll", handleScroll);
      window.removeEventListener("resize", handleScroll);
    }
  });
</script>

<div class="toolbar" class:hidden-at-bottom={isAtBottom}>
  <button
    class="toolbar-button"
    title="Create a new experiment"
    onclick={() => {
      openCreateExperimentModal();
    }}
  >
    <Plus class="icon" />
  </button>

  <button
    class="toolbar-button"
    title="Enter Comparison Mode"
    onclick={() => {
      toggleMode();
    }}
  >
    <GitCompareArrows class="icon" />
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

  <button
    onclick={() => {
      toggleAppTheme();
    }}
    class="toolbar-button"
    aria-label={theme === "dark"
      ? "Switch to light theme"
      : "Switch to dark theme"}
    title={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
  >
    {#if theme === "dark"}
      <Sun class="icon" />
    {:else}
      <Moon class="icon" />
    {/if}
  </button>

  <button
    class="toolbar-button"
    aria-label="go to settings page"
    title="go to settings page"
    onclick={() => {
      goto("/settings");
    }}
  >
    <Cog class="icon" />
  </button>
</div>

<style>
  .toolbar {
    position: fixed;
    bottom: 1rem;
    left: 50vw;
    transform: translateX(-50%) scale(1.25);
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 0;
    background-color: var(--color-ctp-surface1);
    border: 1px solid var(--color-ctp-surface2);
    border-radius: 0.5rem;
    box-shadow:
      0 4px 6px -1px rgba(0, 0, 0, 0.1),
      0 2px 4px -1px rgba(0, 0, 0, 0.06);
    z-index: 40;
    overflow: hidden;
    opacity: 100%;
    transition:
      transform 0.3s ease,
      opacity 0.3s ease;
  }

  .toolbar-button {
    padding: 0.5rem;
    color: var(--color-ctp-subtext0);
    transition:
      color 0.2s,
      background-color 0.2s;
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
      gap: 0;
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
