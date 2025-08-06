<script lang="ts">
  import { goto } from "$app/navigation";
  import Logo from "$lib/logo_assets/logo.svelte";
  import { getTheme, toggleTheme } from "$lib/state/theme.svelte";
  import { Cog, Moon, Sun } from "@lucide/svelte";
  let theme = $derived(getTheme());
</script>

<header class="sticky top-0 z-30 surface-glass-elevated">
  <nav class="px-6 py-4 flex flex-row justify-between items-center">
    <button
      class="w-32 text-ctp-blue fill-current hover:text-ctp-blue/80 transition-colors"
      onclick={() => goto("/")}
    >
      <Logo />
    </button>

    <div class="flex space-x-4">
      <button
        class="floating-element p-2"
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

      <button class="floating-element p-2" onclick={() => goto("/settings")}>
        <Cog />
      </button>
    </div>
  </nav>
</header>
