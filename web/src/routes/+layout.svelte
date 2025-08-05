<script lang="ts">
  import "../app.css";
  import { getTheme } from "$lib/state/theme.svelte.js";
  import Header from "$lib/components/header.svelte";
  import Toolbar from "$lib/components/toolbar.svelte";
  import SimpleLandingPage from "$lib/components/simple-landing-page.svelte";
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

<div class="text-ctp-text surface-base flex-1" data-theme={theme}>
  {#if showNavigation}
    <Header />
    <!-- <Toolbar /> -->
    <main class="layer-fade-in">
      <div class="content-layer">
        {@render children()}
      </div>
    </main>
  {:else}
    <main class="surface-base"><SimpleLandingPage /></main>
  {/if}
</div>
