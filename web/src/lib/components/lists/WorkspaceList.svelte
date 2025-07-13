<script lang="ts">
  import { DeleteWorkspaceModal } from "$lib/components/modals";
  import EmptyState from "./EmptyState.svelte";
  import ListCard from "./ListCard.svelte";
  import WorkspaceRoleBadge from "$lib/components/workspace-role-badge.svelte";
  import { Trash2 } from "@lucide/svelte";
  import type { Workspace } from "$lib/types";

  interface Props {
    workspaces: Workspace[];
    searchQuery?: string;
  }

  let { workspaces, searchQuery = "" }: Props = $props();

  let filteredWorkspaces = $derived(
    workspaces.filter((workspace) => {
      if (!searchQuery) return true;
      const query = searchQuery.toLowerCase();
      const searchableText =
        `${workspace.name} ${workspace.description || ""}`.toLowerCase();

      return query.split(" ").every((term) => searchableText.includes(term));
    }),
  );

  function formatDate(date: Date): string {
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year:
        date.getFullYear() !== new Date().getFullYear() ? "numeric" : undefined,
    });
  }

  function formatTime(date: Date): string {
    return date.toLocaleTimeString("en-US", {
      hour: "numeric",
      minute: "2-digit",
      hour12: true,
    });
  }

  // Workspace deletion state
  let deleteModalOpen = $state(false);
  let workspaceToDelete: Workspace | null = $state(null);

  function openDeleteModal(workspace: Workspace) {
    workspaceToDelete = workspace;
    deleteModalOpen = true;
  }

  function onWorkspaceDeleted() {
    // Refresh the page to update the workspace list
    window.location.reload();
  }

  function canDeleteWorkspace(workspace: Workspace): boolean {
    return workspace.role === "OWNER";
  }

  function getWorkspaceItemClass(workspace: Workspace): string {
    const baseClass = "group layer-slide-up";
    const surfaceClass = "surface-interactive";

    return `${baseClass} ${surfaceClass}`.trim();
  }

  function handleWorkspaceClick(workspace: Workspace) {
    window.location.href = `/workspaces/${workspace.id}`;
  }
</script>

{#if filteredWorkspaces.length === 0 && searchQuery}
  <EmptyState type="search" {searchQuery} />
{:else}
  <ListCard
    items={filteredWorkspaces}
    getItemClass={getWorkspaceItemClass}
    onItemClick={handleWorkspaceClick}
  >
    {#snippet children(workspace)}
      <!-- Content - Mobile-first responsive layout -->
      <div class="flex-1 min-w-0">
        <!-- Header: Name and date -->
        <div
          class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1 sm:gap-3 mb-2"
        >
          <h3
            class="text-ctp-text group-hover:text-ctp-blue transition-colors font-medium truncate"
          >
            {workspace.name}
          </h3>
          <div
            class="flex items-center gap-2 text-xs text-ctp-lavender flex-shrink-0"
          >
            <span>{formatDate(workspace.createdAt)}</span>
            <span class="hidden sm:inline text-ctp-lavender/80"
              >{formatTime(workspace.createdAt)}</span
            >
          </div>
        </div>

        <!-- Description -->
        {#if workspace.description}
          <p
            class="text-ctp-subtext1 text-sm mb-2 line-clamp-2 sm:line-clamp-none"
          >
            {workspace.description}
          </p>
        {/if}

        <!-- Role and metadata - Stack on mobile -->
        <div class="flex items-center gap-2 text-xs">
          <!-- Role badge using the dedicated component -->
          <div class="flex items-center gap-1 flex-wrap">
            <WorkspaceRoleBadge role={workspace.role || "VIEWER"} />
          </div>
        </div>
      </div>
    {/snippet}

    {#snippet actions(workspace)}
      {#if canDeleteWorkspace(workspace)}
        <div class="flex items-center justify-end gap-2">
          <button
            onclick={(e) => {
              e.stopPropagation();
              openDeleteModal(workspace);
            }}
            class="flex items-center gap-1 text-xs text-ctp-subtext0 hover:text-ctp-red transition-colors sm:bg-ctp-surface0/20 sm:backdrop-blur-md sm:border sm:border-ctp-surface0/30 sm:hover:border-ctp-red/30 sm:p-1"
            title="delete workspace"
          >
            <Trash2 class="w-3 h-3" />
            <span>Delete</span>
          </button>
        </div>
      {/if}
    {/snippet}
  </ListCard>
{/if}

<!-- Delete Workspace Modal -->
<DeleteWorkspaceModal
  bind:isOpen={deleteModalOpen}
  workspace={workspaceToDelete}
  onDeleted={onWorkspaceDeleted}
/>
