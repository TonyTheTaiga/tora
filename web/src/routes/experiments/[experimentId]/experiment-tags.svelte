<script lang="ts">
  let { tags, initialLimit = 10 }: { tags: string[]; initialLimit?: number } = $props();
  
  let showAllTags = $state(false);
  
  let visibleTags = $derived(
    showAllTags || tags.length <= initialLimit
      ? tags
      : tags.slice(0, initialLimit),
  );
</script>

{#if tags && tags.length > 0}
  <div class="space-y-2">
    <div class="flex items-center gap-2">
      <div class="text-sm text-ctp-text">tags</div>
      <div class="text-sm text-ctp-subtext0 font-mono">
        [{tags.length}]
      </div>
    </div>
    <div class="flex flex-wrap gap-1">
      {#each visibleTags as tag}
        <span
          class="text-sm bg-ctp-blue/20 text-ctp-blue border border-ctp-blue/30 px-2 py-0.5 font-mono"
        >
          {tag}
        </span>
      {/each}
      {#if tags.length > initialLimit}
        <button
          onclick={() => (showAllTags = !showAllTags)}
          class="text-sm text-ctp-subtext0 hover:text-ctp-blue transition-colors px-2 py-0.5"
        >
          {showAllTags
            ? "less"
            : `+${tags.length - initialLimit}`}
        </button>
      {/if}
    </div>
  </div>
{/if}