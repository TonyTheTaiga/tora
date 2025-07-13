<script lang="ts">
  import "../app.css";
  import { getTheme } from "$lib/state/theme.svelte.js";
  import Header from "$lib/components/header.svelte";
  import Toolbar from "$lib/components/toolbar.svelte";
  import { page } from "$app/state";

  let { children } = $props();

  let theme = $derived.by(() => getTheme());
  let showNavigation = $derived(
    page.url.pathname !== "/" &&
      page.url.pathname !== "/login" &&
      page.url.pathname !== "/signup" &&
      !page.url.pathname.startsWith("/signup/"),
  );
</script>

<div class="min-h-screen text-ctp-text surface-base" data-theme={theme}>
  {#if showNavigation}
    <Header />
    <Toolbar />
    <main
      class="flex-1 w-full max-w-7xl mx-auto px-2 sm:px-4 md:px-6 py-3 sm:py-4 md:py-6 layer-fade-in"
    >
      <div class="content-layer layer-spacing-md">
        {@render children()}
      </div>
    </main>
  {:else}
    <main class="flex-1 w-full surface-base">{@render children()}</main>
  {/if}
</div>
