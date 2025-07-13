<script lang="ts">
  import WorkspaceRoleBadge from "$lib/components/workspace-role-badge.svelte";
  import { DeleteWorkspaceModal } from "$lib/components/modals";
  import EmptyState from "./EmptyState.svelte";
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
</script>

{#if filteredWorkspaces.length === 0 && searchQuery}
  <EmptyState type="search" {searchQuery} />
{:else}
  <!-- Minimal List Layout -->
  <div class="space-y-1 font-mono">
    {#each filteredWorkspaces as workspace}
      <!-- Workspace item with relative positioning for actions -->
      <div
        class="group relative hover:bg-ctp-surface0/10 transition-colors border-l-2 border-transparent hover:border-ctp-blue/30"
      >
        <!-- Main clickable area -->
        <a href={`/workspaces/${workspace.id}`} class="block p-3 md:p-4">
          <div
            class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2"
          >
            <!-- Workspace info -->
            <div class="flex items-center gap-3 min-w-0 flex-1">
              <!-- Name and description -->
              <div class="min-w-0 flex-1">
                <h3
                  class="text-ctp-text group-hover:text-ctp-blue transition-colors font-medium truncate"
                >
                  {workspace.name}
                </h3>
                {#if workspace.description}
                  <p class="text-ctp-subtext1 text-sm mt-1 line-clamp-1">
                    {workspace.description}
                  </p>
                {/if}
              </div>
            </div>

            <!-- Role and date -->
            <div class="flex items-center gap-3 text-xs flex-shrink-0 sm:ml-4">
              <WorkspaceRoleBadge role={workspace.role || "VIEWER"} />
              <span class="text-ctp-subtext0">
                {formatDate(workspace.createdAt)}
              </span>
            </div>
          </div>
        </a>

        <!-- Delete button for owned workspaces -->
        {#if canDeleteWorkspace(workspace)}
          <div class="border-t border-ctp-surface0/20 px-3 py-2">
            <div class="flex items-center justify-end">
              <button
                onclick={(e) => {
                  e.stopPropagation();
                  openDeleteModal(workspace);
                }}
                class="flex items-center gap-1 text-xs text-ctp-subtext0 hover:text-ctp-red transition-colors sm:bg-ctp-surface0/20 sm:backdrop-blur-md sm:border sm:border-ctp-surface0/30 sm:hover:border-ctp-red/30 sm:p-1"
                title="delete workspace"
              >
                <Trash2 class="w-3 h-3" />
                <span class="sm:hidden">Delete</span>
              </button>
            </div>
          </div>
        {/if}
      </div>
    {/each}
  </div>
{/if}

<!-- Delete Workspace Modal -->
<DeleteWorkspaceModal
  bind:isOpen={deleteModalOpen}
  workspace={workspaceToDelete}
  onDeleted={onWorkspaceDeleted}
/>
