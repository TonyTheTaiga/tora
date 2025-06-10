<script lang="ts">
  import {
    Plus,
    User,
    Briefcase,
    Moon,
    Sun,
    GitCompareArrows,
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
</div>
