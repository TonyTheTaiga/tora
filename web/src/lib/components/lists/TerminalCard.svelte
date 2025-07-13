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

<div class="space-y-3 font-mono">
  {#each items as item, index}
    <div class={getItemClass(item)} style="animation-delay: {index * 50}ms">
      <div class="surface-interactive layer-slide-up">
        <!-- Header row with icon, name, metadata, and date -->
        <div class="terminal-chrome-header">
          <div class="flex items-center gap-2 min-w-0 flex-1">
            {@render children(item)}
          </div>
        </div>

        <!-- Description row (if any) -->
        {#if item.description}
          <div class="px-4 py-3 text-ctp-subtext1 text-sm surface-layer-1">
            {item.description}
          </div>
        {/if}

        <!-- Actions row (if any) -->
        {#if actions}
          <div class="px-4 py-3 surface-layer-2 terminal-border-accent">
            {@render actions(item)}
          </div>
        {/if}
      </div>
    </div>
  {/each}
</div>
