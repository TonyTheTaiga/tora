<script lang="ts">
  import type { Snippet } from "svelte";

  interface Props {
    items: any[];
    onItemClick?: (item: any) => void;
    children: Snippet<[any]>;
    actions?: Snippet<[any]>;
  }

  let { items, onItemClick, children, actions }: Props = $props();

  function handleItemClick(item: any) {
    onItemClick?.(item);
  }
</script>

<div role="list" class="font-mono">
  {#each items as item}
    <div role="listitem" class="group layer-slide-up">
      <div class="flex items-start justify-between gap-4 px-2 py-2">
        <button
          onclick={() => handleItemClick(item)}
          class="text-left flex-1 min-w-0 focus:outline-none"
        >
          {@render children(item)}
        </button>
        {#if actions}
          {@render actions(item)}
        {/if}
      </div>
    </div>
  {/each}
  {#if !items || items.length === 0}
    <div class="text-ctp-subtext1 text-sm">no items</div>
  {/if}
</div>
