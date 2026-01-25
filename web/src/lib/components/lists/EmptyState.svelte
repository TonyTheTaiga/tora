<script lang="ts">
  import { Search, FolderOpen, FileQuestion, Plus } from "@lucide/svelte";

  interface Props {
    type: "search" | "empty";
    searchQuery?: string;
    itemType?: string; // e.g., "workspaces", "experiments"
    actionText?: string;
    onAction?: () => void;
  }

  let {
    type,
    searchQuery = "",
    itemType = "items",
    actionText,
    onAction,
  }: Props = $props();

  // Map item types to icons
  const emptyIcons: Record<string, typeof FolderOpen> = {
    workspaces: FolderOpen,
    experiments: FileQuestion,
  };

  let EmptyIcon = $derived(emptyIcons[itemType] ?? FileQuestion);
</script>

{#if type === "search"}
  <!-- No search results -->
  <div class="flex flex-col items-center justify-center py-8 text-center">
    <div class="p-4 bg-ctp-surface0/20 border border-ctp-surface0/30 mb-4">
      <Search size={28} class="text-ctp-subtext0" />
    </div>
    <div class="text-ctp-subtext0 text-sm">
      no results for "<span class="text-ctp-text">{searchQuery}</span>"
    </div>
    <p class="text-ctp-subtext1 text-xs mt-1">try a different search term</p>
  </div>
{:else if type === "empty"}
  <!-- Empty state -->
  <div class="flex flex-col items-center justify-center py-8 text-center">
    <div class="p-4 bg-ctp-surface0/20 border border-ctp-surface0/30 mb-4">
      <EmptyIcon size={28} class="text-ctp-subtext0" />
    </div>
    <div class="text-ctp-subtext0 text-sm">
      no {itemType} found
    </div>
    {#if actionText && onAction}
      <button
        onclick={onAction}
        class="mt-4 flex items-center gap-2 text-ctp-blue hover:text-ctp-blue/80 transition-colors text-sm"
      >
        <Plus size={16} />
        {actionText}
      </button>
    {/if}
  </div>
{/if}
