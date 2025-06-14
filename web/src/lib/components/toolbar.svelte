<script lang="ts">
  import {
    Plus,
    Moon,
    Sun,
    GitCompareArrows,
    Cog,
    X,
    ArrowRight,
    ArrowLeft,
    Command,
  } from "lucide-svelte";
  import { goto } from "$app/navigation";
  import { onMount, onDestroy } from "svelte";
  import {
    toggleMode,
    getMode,
    getExperimentsSelectedForComparision,
  } from "$lib/state/comparison.svelte.js";
  import {
    getTheme,
    toggleTheme as toggleAppTheme,
  } from "$lib/state/theme.svelte.js";
  import { openCreateExperimentModal } from "$lib/state/app.svelte.js";
  import { page } from "$app/state";

  let theme = $derived.by(() => getTheme());
  let isComparisonMode = $derived.by(() => getMode());
  let selectedExperiments = $derived.by(() =>
    getExperimentsSelectedForComparision(),
  );
  let isWorkspacePage = $derived(page.url.pathname.startsWith("/workspaces/"));
  let showBackButton = $derived.by(() => {
    const path = page.url.pathname;
    return path !== '/' && 
           path !== '/workspaces' &&
           !path.startsWith('/login') && 
           !path.startsWith('/signup');
  });

  let visible = $state(true);
  let lastScrollY = $state(0);
  const scrollThreshold = 10;

  const handleScroll = () => {
    if (typeof window !== "undefined" && typeof document !== "undefined") {
      const currentScrollY = window.scrollY;
      const documentHeight = document.documentElement.scrollHeight;
      const windowHeight = window.innerHeight;
      const hasScroll = documentHeight > windowHeight;

      if (!hasScroll) {
        visible = true;
        lastScrollY = Math.max(0, currentScrollY); // Clamp to 0 to ignore rubber banding
        return;
      }

      const atBottom = windowHeight + currentScrollY >= documentHeight - 1;

      if (atBottom) {
        visible = false;
        lastScrollY = currentScrollY;
        return;
      }

      // At top or during rubber band effect (negative scroll)
      if (currentScrollY <= 0) {
        visible = true;
        lastScrollY = Math.max(0, currentScrollY); // Clamp to 0 to ignore rubber banding
        return;
      }

      if (Math.abs(currentScrollY - lastScrollY) > scrollThreshold) {
        visible = currentScrollY < lastScrollY;
      }

      lastScrollY = currentScrollY;
    }
  };

  onMount(() => {
    if (typeof window !== "undefined") {
      lastScrollY = window.scrollY;
      window.addEventListener("scroll", handleScroll, { passive: true });
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

<div
  class="fixed bottom-4 left-1/2 -translate-x-1/2 z-40 transition-all duration-300 {visible
    ? 'opacity-100'
    : 'opacity-0 translate-y-full pointer-events-none'}"
>
  <div
    class="flex items-center bg-ctp-surface1/30 backdrop-blur-xl border border-ctp-surface0/20 rounded-full p-1 shadow-2xl hover:bg-ctp-surface1/40 hover:scale-105 transition-all duration-200"
  >
    <style>
      @keyframes slideInRight {
        from {
          opacity: 0;
          transform: translateX(20px);
        }
        to {
          opacity: 1;
          transform: translateX(0);
        }
      }

      @keyframes fadeIn {
        from {
          opacity: 0;
        }
        to {
          opacity: 1;
        }
      }
    </style>
    {#if showBackButton}
      <button
        class="p-3 rounded-full hover:bg-ctp-surface0/50 transition-all duration-200 text-ctp-subtext0 hover:text-ctp-text hover:scale-110 active:scale-95"
        title="Go back"
        onclick={() => history.back()}
      >
        <ArrowLeft size={20} />
      </button>
    {/if}

    <button
      class="p-3 rounded-full hover:bg-ctp-surface0/50 transition-all duration-200 text-ctp-subtext0 hover:text-ctp-text hover:scale-110 active:scale-95"
      title="Go to workspaces"
      onclick={() => {
        goto("/");
      }}
    >
      <Command size={20} />
    </button>

    {#if isWorkspacePage}
      <button
        class="p-3 rounded-full hover:bg-ctp-surface0/50 transition-all duration-200 text-ctp-subtext0 hover:text-ctp-text hover:scale-110 active:scale-95"
        title="Create a new experiment"
        onclick={() => {
          openCreateExperimentModal();
        }}
      >
        <Plus size={20} />
      </button>

      {#if !isComparisonMode}
        <button
          class="p-3 rounded-full hover:bg-ctp-surface0/50 transition-all duration-200 text-ctp-subtext0 hover:text-ctp-text hover:scale-110 active:scale-95"
          title="Enter Comparison Mode"
          onclick={() => {
            toggleMode();
          }}
        >
          <GitCompareArrows size={20} />
        </button>
      {:else}
        <div
          class="flex items-center"
          style="animation: slideInRight 0.3s ease-out;"
        >
          <span
            class="text-xs text-ctp-subtext0 px-2 whitespace-nowrap"
            style="animation: fadeIn 0.5s ease-out 0.1s both;"
          >
            {selectedExperiments.length} selected
          </span>
          <button
            class="p-2 rounded-full hover:bg-ctp-red/20 transition-all duration-200 text-ctp-red hover:text-ctp-red hover:scale-110 active:scale-95"
            style="animation: slideInRight 0.3s ease-out 0.15s both;"
            title="Cancel Comparison"
            onclick={() => {
              toggleMode();
            }}
          >
            <X size={18} />
          </button>
          <button
            class="p-2 rounded-full hover:bg-ctp-blue/20 transition-all duration-200 text-ctp-blue hover:text-ctp-blue hover:scale-110 active:scale-95 {selectedExperiments.length <
            2
              ? 'opacity-50 cursor-not-allowed'
              : ''}"
            style="animation: slideInRight 0.3s ease-out 0.2s both;"
            title="Compare Selected"
            disabled={selectedExperiments.length < 2}
            onclick={() => {
              if (selectedExperiments.length >= 2) {
                const params = selectedExperiments.join(",");
                goto(`/compare?ids=${params}`);
              }
            }}
          >
            <ArrowRight size={18} />
          </button>
        </div>
      {/if}
    {/if}

    <button
      onclick={() => {
        toggleAppTheme();
      }}
      class="p-3 rounded-full hover:bg-ctp-surface0/50 transition-all duration-200 text-ctp-subtext0 hover:text-ctp-text hover:scale-110 active:scale-95"
      aria-label={theme === "dark"
        ? "Switch to light theme"
        : "Switch to dark theme"}
      title={theme === "dark"
        ? "Switch to light theme"
        : "Switch to dark theme"}
    >
      {#if theme === "dark"}
        <Sun size={20} />
      {:else}
        <Moon size={20} />
      {/if}
    </button>

    <button
      class="p-3 rounded-full hover:bg-ctp-surface0/50 transition-all duration-200 text-ctp-subtext0 hover:text-ctp-text hover:scale-110 active:scale-95"
      aria-label="go to settings page"
      title="go to settings page"
      onclick={() => {
        goto("/settings");
      }}
    >
      <Cog size={20} />
    </button>
  </div>
</div>
