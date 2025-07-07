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

<div class="min-h-screen bg-ctp-base text-ctp-text" data-theme={theme}>
  {#if showNavigation}
    <Header />
    <Toolbar />
    <main class="flex-1 w-full max-w-7xl mx-auto p-2">
      {@render children()}
    </main>
  {:else}
    <main class="flex-1 w-full">{@render children()}</main>
  {/if}
</div>
