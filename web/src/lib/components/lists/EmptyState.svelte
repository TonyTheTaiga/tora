<script lang="ts">
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
</script>

{#if type === "search"}
  <!-- No search results -->
  <div class="text-ctp-subtext0 text-base">
    <div>search "{searchQuery}"</div>
    <div class="text-ctp-subtext1 ml-2">no results found</div>
  </div>
{:else if type === "empty"}
  <!-- Empty state -->
  <div class="space-y-3 text-base">
    <div class="text-ctp-subtext0 text-sm">
      no {itemType} found
    </div>
    {#if actionText && onAction}
      <div class="mt-4">
        <button
          onclick={onAction}
          class="text-ctp-blue hover:text-ctp-blue/80 transition-colors text-sm"
        >
          [{actionText}]
        </button>
      </div>
    {/if}
  </div>
{/if}
