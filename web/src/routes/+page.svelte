<script lang="ts">
  import LandingPage from "$lib/components/landing-page.svelte";
  import type { PageData } from "./$types";

  let { data }: { data: PageData } = $props();
</script>

<LandingPage />
