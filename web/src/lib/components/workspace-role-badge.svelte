<script lang="ts">
  let { role }: { role: string } = $props();
</script>

{#if role === "OWNER"}
  <span class="text-ctp-yellow font-medium flex-shrink-0 text-xs font-mono"
    >OWNER</span
  >
{:else if role === "VIEWER"}
  <span class="text-ctp-lavender font-medium flex-shrink-0 text-xs font-mono"
    >VIEWER</span
  >
{/if}
