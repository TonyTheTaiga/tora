<script lang="ts">
  import { page } from "$app/state";
</script>

<div class="font-mono text-ctp-text p-4">
  <div class="space-y-4">
    <div>
      {#if page && page.error}
        {page.error.message}
      {:else}
        something went horribly wrong
      {/if}
    </div>
    
    <a 
      href="/" 
      class="text-ctp-blue hover:text-ctp-blue/80 transition-colors text-sm"
    >
      [back to workspace]
    </a>
  </div>
</div>
