<script lang="ts">
  import LandingPage from "$lib/components/landing-page.svelte";

  interface PageData {
    highlightedCodeDark: string;
    highlightedCodeLight: string;
    processedUserGuide: string;
  }

  let { data }: { data: PageData } = $props();
</script>

<LandingPage
  highlightedCodeDark={data.highlightedCodeDark}
  highlightedCodeLight={data.highlightedCodeLight}
  processedUserGuide={data.processedUserGuide}
/>
