<script lang="ts">
  let { role }: { role: string } = $props();
</script>

{#if role === "OWNER"}
  <span
    class="text-sm px-2 py-0.5 bg-ctp-yellow/20 text-ctp-yellow border border-ctp-yellow/30 rounded-full font-medium flex-shrink-0"
  >
    OWNER
  </span>
{:else if role === "ADMIN"}
  <span
    class="text-sm px-2 py-0.5 bg-ctp-red/20 text-ctp-red border border-ctp-red/30 rounded-full font-medium flex-shrink-0"
  >
    ADMIN
  </span>
{:else if role === "EDITOR"}
  <span
    class="text-sm px-2 py-0.5 bg-ctp-blue/20 text-ctp-blue border border-ctp-blue/30 rounded-full font-medium flex-shrink-0"
  >
    EDITOR
  </span>
{:else if role === "VIEWER"}
  <span
    class="text-sm px-2 py-0.5 bg-ctp-green/20 text-ctp-green border border-ctp-green/30 rounded-full font-medium flex-shrink-0"
  >
    VIEWER
  </span>
{/if}
