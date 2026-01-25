<script lang="ts">
  let { role }: { role: string } = $props();
</script>

{#if role === "OWNER"}
  <span class="text-[11px] font-mono flex-shrink-0 text-ctp-subtext0"
    ><span class="text-ctp-overlay0">role:</span>owner</span
  >
{:else if role === "VIEWER"}
  <span class="text-[11px] font-mono flex-shrink-0 text-ctp-subtext0"
    ><span class="text-ctp-overlay0">role:</span>viewer</span
  >
{/if}
