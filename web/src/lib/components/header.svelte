<script lang="ts">
  import { goto } from "$app/navigation";
  import Logo from "$lib/logo_assets/logo.svelte";
  import { getTheme, toggleTheme } from "$lib/state/theme.svelte";
  import { Cog, Moon, Sun, House, Globe } from "@lucide/svelte";
  let theme = $derived(getTheme());
</script>

<header class="shrink-0 sticky top-0 z-30 surface-glass-elevated">
  <nav class="px-6 py-4 flex flex-row justify-between items-center">
    <div class="w-32 text-ctp-blue fill-current">
      <Logo />
    </div>

    <div class="flex space-x-4">
      <button class="floating-element p-2 rounded-md">
        <House size={20} onclick={() => goto("/")} />
      </button>

      <button class="floating-element p-2 rounded-md">
        <Globe size={20} />
      </button>

      <button
        class="floating-element p-2 rounded-md"
        onclick={() => {
          toggleTheme();
        }}
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
        class="floating-element p-2 rounded-md"
        onclick={() => goto("/settings")}
      >
        <Cog />
      </button>
    </div>
  </nav>
</header>
