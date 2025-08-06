<script lang="ts">
  import {
    DeleteWorkspaceModal,
    WorkspaceInviteModal,
  } from "$lib/components/modals";
  import EmptyState from "./EmptyState.svelte";
  import ListCard from "./ListCard.svelte";
  import WorkspaceRoleBadge from "$lib/components/workspace-role-badge.svelte";
  import { Trash2, Users, LogOut } from "@lucide/svelte";
  import type { Workspace } from "$lib/types";

  interface Props {
    workspaces: Workspace[];
    searchQuery: string;
    workspaceRoles: Array<{ id: string; name: string }>;
    onItemClick: (workspace: Workspace) => void;
  }

  let { workspaces, searchQuery, workspaceRoles, onItemClick }: Props =
    $props();

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

  let deleteModalOpen = $state(false);
  let workspaceToDelete: Workspace | null = $state(null);
  let inviteModalOpen = $state(false);
  let workspaceToInvite: Workspace | null = $state(null);

  function openDeleteModal(workspace: Workspace) {
    workspaceToDelete = workspace;
    deleteModalOpen = true;
  }

  function openInviteModal(workspace: Workspace) {
    workspaceToInvite = workspace;
    inviteModalOpen = true;
  }

  function onWorkspaceDeleted() {
    // Refresh the page to update the workspace list
    window.location.reload();
  }

  function canDeleteWorkspace(workspace: Workspace): boolean {
    return workspace.role === "OWNER";
  }

  function canInviteToWorkspace(workspace: Workspace): boolean {
    return workspace.role === "OWNER";
  }

  function canLeaveWorkspace(workspace: Workspace): boolean {
    return workspace.role !== "OWNER";
  }

  function sendInvitation(email: string, roleId: string) {
    if (!workspaceToInvite) return;

    const form = document.createElement("form");
    form.method = "POST";
    form.action = "?/sendInvitation";

    const workspaceIdInput = document.createElement("input");
    workspaceIdInput.type = "hidden";
    workspaceIdInput.name = "workspaceId";
    workspaceIdInput.value = workspaceToInvite.id;
    form.appendChild(workspaceIdInput);

    const emailInput = document.createElement("input");
    emailInput.type = "hidden";
    emailInput.name = "email";
    emailInput.value = email;
    form.appendChild(emailInput);

    const roleIdInput = document.createElement("input");
    roleIdInput.type = "hidden";
    roleIdInput.name = "roleId";
    roleIdInput.value = roleId;
    form.appendChild(roleIdInput);

    document.body.appendChild(form);
    form.submit();

    inviteModalOpen = false;
    workspaceToInvite = null;
  }

  function leaveWorkspace(workspaceId: string) {
    if (!confirm("Are you sure you want to leave this workspace?")) return;

    const form = document.createElement("form");
    form.method = "POST";
    form.action = "?/leaveWorkspace";

    const workspaceIdInput = document.createElement("input");
    workspaceIdInput.type = "hidden";
    workspaceIdInput.name = "workspaceId";
    workspaceIdInput.value = workspaceId;
    form.appendChild(workspaceIdInput);

    document.body.appendChild(form);
    form.submit();
  }

  function getWorkspaceItemClass(): string {
    const baseClass = "group layer-slide-up";
    const surfaceClass = "surface-interactive";

    return `${baseClass} ${surfaceClass}`.trim();
  }
</script>

{#if filteredWorkspaces.length === 0 && searchQuery}
  <EmptyState type="search" {searchQuery} />
{:else}
  <ListCard
    items={filteredWorkspaces}
    getItemClass={getWorkspaceItemClass}
    {onItemClick}
  >
    {#snippet children(workspace)}
      <div class="flex-1 min-w-0">
        <div class="flex items-center justify-between gap-3 mb-2">
          <h3
            class="text-ctp-text group-hover:text-ctp-blue transition-colors font-medium truncate"
          >
            {workspace.name}
          </h3>
          <div
            class="flex items-center gap-2 text-xs text-ctp-lavender flex-shrink-0"
          >
            <span>{formatDate(workspace.createdAt)}</span>
            <span class="text-ctp-lavender/80"
              >{formatTime(workspace.createdAt)}</span
            >
          </div>
        </div>

        {#if workspace.description}
          <p class="text-ctp-subtext1 text-sm mb-2">
            {workspace.description}
          </p>
        {/if}

        <div class="flex items-center gap-2 text-xs">
          <div class="flex items-center gap-1 flex-wrap">
            <WorkspaceRoleBadge role={workspace.role || "VIEWER"} />
          </div>
        </div>
      </div>
    {/snippet}

    {#snippet actions(workspace)}
      <div class="flex items-center justify-end gap-2">
        {#if canInviteToWorkspace(workspace)}
          <button
            onclick={(e) => {
              e.stopPropagation();
              openInviteModal(workspace);
            }}
            class="flex items-center gap-1 text-xs text-ctp-subtext0 hover:text-ctp-blue transition-colors bg-ctp-surface0/20 backdrop-blur-md border border-ctp-surface0/30 hover:border-ctp-blue/30 p-1"
            title="invite users"
          >
            <Users class="w-3 h-3" />
            <span>Invite</span>
          </button>
        {/if}
        {#if canDeleteWorkspace(workspace)}
          <button
            onclick={(e) => {
              e.stopPropagation();
              openDeleteModal(workspace);
            }}
            class="flex items-center gap-1 text-xs text-ctp-subtext0 hover:text-ctp-red transition-colors bg-ctp-surface0/20 backdrop-blur-md border border-ctp-surface0/30 hover:border-ctp-red/30 p-1"
            title="delete workspace"
          >
            <Trash2 class="w-3 h-3" />
            <span>Delete</span>
          </button>
        {/if}
        {#if canLeaveWorkspace(workspace)}
          <button
            onclick={(e) => {
              e.stopPropagation();
              leaveWorkspace(workspace.id);
            }}
            class="flex items-center gap-1 text-xs text-ctp-subtext0 hover:text-ctp-red transition-colors bg-ctp-surface0/20 backdrop-blur-md border border-ctp-surface0/30 hover:border-ctp-red/30 p-1"
            title="leave workspace"
          >
            <LogOut class="w-3 h-3" />
            <span>Leave</span>
          </button>
        {/if}
      </div>
    {/snippet}
  </ListCard>
{/if}

<!-- Delete Workspace Modal -->
<DeleteWorkspaceModal
  bind:isOpen={deleteModalOpen}
  workspace={workspaceToDelete}
  onDeleted={onWorkspaceDeleted}
/>

<!-- Workspace Invite Modal -->
<WorkspaceInviteModal
  bind:isOpen={inviteModalOpen}
  workspace={workspaceToInvite}
  {workspaceRoles}
  onInvite={sendInvitation}
/>
