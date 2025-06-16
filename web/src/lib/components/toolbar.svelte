<script lang="ts">
  import { Moon, Sun, ArrowLeft, Command, Cog } from "lucide-svelte";
  import { goto } from "$app/navigation";
  import { onMount, onDestroy } from "svelte";
  import {
    getTheme,
    toggleTheme as toggleAppTheme,
  } from "$lib/state/theme.svelte.js";
  import { getToolbarButtons } from "$lib/state/toolbar.svelte.js";
  import { page } from "$app/state";

  let theme = $derived.by(() => getTheme());

  let showBackButton = $derived.by(() => {
    const path = page.url.pathname;
    return (
      path !== "/" &&
      path !== "/workspaces" &&
      !path.startsWith("/login") &&
      !path.startsWith("/signup")
    );
  });

  let dynamicButtons = $derived.by(() => getToolbarButtons());

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



    {#each dynamicButtons as btn (btn.id)}
      {@const Icon = btn.icon}
      <button
        class="p-3 rounded-full hover:bg-ctp-surface0/50 transition-all duration-200 text-ctp-subtext0 hover:text-ctp-text hover:scale-110 active:scale-95"
        aria-label={btn.ariaLabel}
        title={btn.title}
        onclick={() => {
          btn.onClick();
        }}
      >
        <Icon size={20} />
      </button>
    {/each}

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
