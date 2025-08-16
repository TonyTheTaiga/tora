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

{#if showNavigation}
  <main class="flex flex-col h-full w-full overflow-hidden">
    <Header />
    {@render children()}
  </main>
{:else}
  <main class="h-full w-full">
    {@render children()}
  </main>
{/if}
