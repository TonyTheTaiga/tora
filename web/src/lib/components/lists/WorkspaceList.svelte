<script lang="ts">
  import WorkspaceRoleBadge from "$lib/components/workspace-role-badge.svelte";
  import TerminalCard from "./TerminalCard.svelte";
  import EmptyState from "./EmptyState.svelte";
  import type { Workspace } from "$lib/types";

  interface Props {
    workspaces: Workspace[];
    searchQuery?: string;
    showSummary?: boolean;
  }

  let { workspaces, searchQuery = "", showSummary = true }: Props = $props();

  let filteredWorkspaces = $derived(
    workspaces.filter((workspace) => {
      if (!searchQuery) return true;
      const query = searchQuery.toLowerCase();
      const searchableText =
        `${workspace.name} ${workspace.description || ""}`.toLowerCase();

      return query.split(" ").every((term) => searchableText.includes(term));
    }),
  );
</script>

{#if filteredWorkspaces.length === 0 && searchQuery}
  <EmptyState type="search" {searchQuery} />
{:else}
  <TerminalCard items={filteredWorkspaces}>
    {#snippet children(workspace)}
      <a
        href={`/workspaces/${workspace.id}`}
        class="flex items-center justify-between min-w-0 flex-1 group"
      >
        <span
          class="text-ctp-text group-hover:text-ctp-blue transition-colors font-medium truncate"
        >
          {workspace.name}
        </span>
        <div class="flex items-center gap-2 text-xs text-ctp-subtext0">
          <WorkspaceRoleBadge role={workspace.role || "VIEWER"} />
          <span>|</span>
          <span>
            {new Date(workspace.createdAt).toLocaleDateString("en-US", {
              month: "short",
              day: "numeric",
            })}
          </span>
        </div>
      </a>
    {/snippet}
  </TerminalCard>

  {#if showSummary}
    <div
      class="flex items-center text-sm text-ctp-subtext0 pt-2 border-t border-ctp-surface0/20 mt-4"
    >
      <div class="flex-1">
        {filteredWorkspaces.length} workspace{filteredWorkspaces.length !== 1
          ? "s"
          : ""} total
      </div>
    </div>
  {/if}
{/if}
