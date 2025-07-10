<script lang="ts">
  let { role }: { role: string } = $props();
</script>

{#if role === "OWNER"}
  <span class="text-ctp-yellow font-medium flex-shrink-0">OWNER</span>
{:else if role === "ADMIN"}
  <span class="text-ctp-red font-medium flex-shrink-0">ADMIN</span>
{:else if role === "EDITOR"}
  <span class="text-ctp-blue font-medium flex-shrink-0">EDITOR</span>
{:else if role === "VIEWER"}
  <span class="text-ctp-green font-medium flex-shrink-0">VIEWER</span>
{/if}
