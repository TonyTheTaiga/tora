<script lang="ts">
  import "../app.css";
  import Header from "$lib/components/header.svelte";
  import { page } from "$app/state";

  let { children } = $props();

  let showNavigation = $derived(
    page.url.pathname !== "/" &&
      page.url.pathname !== "/login" &&
      page.url.pathname !== "/signup" &&
      !page.url.pathname.startsWith("/signup/"),
  );
</script>

<div class="text-ctp-text surface-base flex-1">
  {#if showNavigation}
    <Header />
    <!-- <Toolbar /> -->
    <main class="layer-fade-in">
      {@render children()}
    </main>
  {:else}
    <main
      class="surface-base min-h-screen flex flex-col justify-center items-center"
    >
      {@render children()}
    </main>
  {/if}
</div>
