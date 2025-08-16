<script lang="ts">
  import LandingPageV2 from "$lib/components/landing-page-v2.svelte";

  interface PageData {
    highlightedCode: string;
    processedUserGuide: string;
  }

  let { data }: { data: PageData } = $props();
</script>

<LandingPageV2
  highlightedCode={data.highlightedCode}
  processedUserGuide={data.processedUserGuide}
/>
