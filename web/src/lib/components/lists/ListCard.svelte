<script lang="ts">
  import type { Snippet } from "svelte";

  interface Props {
    items: any[];
    getItemClass?: (item: any) => string;
    onItemClick?: (item: any) => void;
    children: Snippet<[any]>;
    actions?: Snippet<[any]>;
  }

  let {
    items,
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

<div class="space-y-1 font-mono">
  {#each items as item}
    <div class={getItemClass(item)}>
      <button onclick={() => handleItemClick(item)} class="w-full text-left">
        <div class="p-4">
          {@render children(item)}
        </div>
      </button>

      {#if actions}
        <div class="border-t border-ctp-surface0/20 px-3 py-2">
          {@render actions(item)}
        </div>
      {/if}
    </div>
  {/each}
</div>
