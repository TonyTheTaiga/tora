<script lang="ts">
  import type { Snippet } from "svelte";

  interface Props {
    items: any[];
    selectedItems?: string[];
    getItemClass?: (item: any) => string;
    onItemClick?: (item: any) => void;
    children: Snippet<[any]>;
    actions?: Snippet<[any]>;
  }

  let {
    items,
    selectedItems = [],
    getItemClass = () =>
      "group transition-colors cursor-pointer hover:bg-ctp-surface0/5 relative mb-1 border border-transparent hover:border-ctp-surface0/8 border-l-2 hover:border-l-ctp-blue/30",
    onItemClick,
    children,
    actions,
  }: Props = $props();

  function handleItemClick(item: any) {
    if (onItemClick) {
      onItemClick(item);
    }
  }
</script>

<!-- Clean List Layout -->
<div class="space-y-1 font-mono">
  {#each items as item}
    <!-- Clean item with relative positioning for actions -->
    <div class={getItemClass(item)}>
      <!-- Main clickable area -->
      <button onclick={() => handleItemClick(item)} class="w-full text-left">
        <div class="p-2 sm:p-3 md:p-4">
          {@render children(item)}
        </div>
      </button>

      <!-- Action buttons - Bottom row for both mobile and desktop -->
      {#if actions}
        <div class="border-t border-ctp-surface0/20 px-2 sm:px-3 py-2">
          {@render actions(item)}
        </div>
      {/if}
    </div>
  {/each}
</div>
