<script lang="ts">
  import type { Snippet } from "svelte";

  interface Props {
    items: any[];
    selectedItems?: string[];
    children: Snippet<[any]>;
    actions?: Snippet<[any]>;
  }

  let { items, selectedItems = [], children, actions }: Props = $props();

  function getItemClass(item: any): string {
    const baseClass = "group transition-colors";
    const selectedClass = selectedItems.includes(item.id)
      ? "bg-ctp-blue/10 border-l-2 border-ctp-blue"
      : "";

    return `${baseClass} ${selectedClass}`.trim();
  }
</script>

<div class="space-y-2 font-mono">
  {#each items as item}
    <div class={getItemClass(item)}>
      <div
        class="bg-ctp-surface0/10 backdrop-blur-md border border-ctp-surface0/20 hover:bg-ctp-surface0/20 transition-colors"
      >
        <!-- Header row with icon, name, metadata, and date -->
        <div
          class="flex items-center justify-between p-3 border-b border-ctp-surface0/20"
        >
          <div class="flex items-center gap-2 min-w-0 flex-1">
            {@render children(item)}
          </div>
        </div>

        <!-- Description row (if any) -->
        {#if item.description}
          <div class="px-3 py-2 text-ctp-subtext1 text-sm">
            {item.description}
          </div>
        {/if}

        <!-- Actions row (if any) -->
        {#if actions}
          <div class="px-3 py-2 border-t border-ctp-surface0/20">
            {@render actions(item)}
          </div>
        {/if}
      </div>
    </div>
  {/each}
</div>
