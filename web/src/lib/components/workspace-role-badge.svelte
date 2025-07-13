<script lang="ts">
  let { role }: { role: string } = $props();
</script>

{#if role === "OWNER"}
  <span class="text-ctp-yellow font-medium flex-shrink-0 text-xs font-mono"
    >OWNER</span
  >
{:else if role === "ADMIN"}
  <span class="text-ctp-red font-medium flex-shrink-0 text-xs font-mono"
    >ADMIN</span
  >
{:else if role === "EDITOR"}
  <span class="text-ctp-blue font-medium flex-shrink-0 text-xs font-mono"
    >EDITOR</span
  >
{:else if role === "VIEWER"}
  <span class="text-ctp-green font-medium flex-shrink-0 text-xs font-mono"
    >VIEWER</span
  >
{/if}
