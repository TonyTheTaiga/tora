<script lang="ts">
  import LandingPage from "$lib/components/landing-page.svelte";
  import type { PageData } from "./$types";

  let { data }: { data: PageData } = $props();
  let { user } = $derived(data);
</script>

{#if !user}
  <LandingPage />
{/if}
