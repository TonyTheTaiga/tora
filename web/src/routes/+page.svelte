<script lang="ts">
  import { LandingPageV2 } from "$lib/components";

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
