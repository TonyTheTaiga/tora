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
  <div class="flex flex-col h-full">
    <Header />
    {@render children()}
  </div>
{:else}
  <main class="h-full w-full">
    {@render children()}
  </main>
{/if}
