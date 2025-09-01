<script lang="ts">
  import {
    openCreateWorkspaceModal,
    getCreateWorkspaceModal,
  } from "$lib/state/modal.svelte";
  import { setSelectedWorkspace } from "./state.svelte";
  import WorkspaceList from "$lib/components/lists/WorkspaceList.svelte";
  import CreateWorkspaceModal from "$lib/components/modals/create-workspace-modal.svelte";
  import { Mail, FolderPlus } from "@lucide/svelte";
  import InvitationsModal from "./invitations-modal.svelte";

  let { workspaces, workspaceRoles, workspaceInvitations } = $props();
  let workspaceSearchQuery = $state("");
  let createWorkspaceModal = $derived(getCreateWorkspaceModal());
  let openInvitationModal = $state<boolean>(false);
</script>

{#if createWorkspaceModal}
  <CreateWorkspaceModal />
{/if}

{#if openInvitationModal}
  <InvitationsModal
    invitations={workspaceInvitations}
    close={() => {
      openInvitationModal = false;
    }}
  />
{/if}

<div class="flex flex-col">
  <div
    class="sticky top-0 z-10 surface-elevated border-b border-ctp-surface0/30 p-4 min-w-0"
  >
    <div class="flex items-center justify-between mb-3">
      <h2 class="text-ctp-text font-medium text-base">Workspaces</h2>
      <div class="flex gap-2">
        <button
          aria-label="create-workspace"
          onclick={() => openCreateWorkspaceModal()}
          class="floating-element p-2"
        >
          <FolderPlus size={16} />
        </button>

        <button
          aria-label="pending-invitations"
          class="floating-element p-2 relative"
          onclick={() => (openInvitationModal = true)}
        >
          <Mail size={16} />
          {#if workspaceInvitations.length > 0}
            <div
              class="absolute -top-1 -right-1 w-3 h-3 bg-ctp-red rounded-full animate-pulse"
            ></div>
          {/if}
        </button>
      </div>
    </div>

    <div
      class="flex items-center bg-ctp-surface0/30 focus-within:ring-1 focus-within:ring-ctp-blue/30 transition-all mb-3 border border-ctp-surface0/40 min-w-0 overflow-hidden"
    >
      <input
        type="search"
        bind:value={workspaceSearchQuery}
        placeholder="search workspaces..."
        class="flex-1 w-full min-w-0 bg-transparent border-0 py-2 px-3 text-ctp-text placeholder-ctp-subtext0 focus:outline-none text-sm truncate"
      />
    </div>
  </div>

  <!-- Content Area -->
  <div class="p-4">
    {#if workspaces.length === 0}
      <div class="text-center py-8 text-ctp-subtext0 text-sm">
        no workspaces found
      </div>
    {:else}
      <WorkspaceList
        {workspaces}
        searchQuery={workspaceSearchQuery}
        onItemClick={setSelectedWorkspace}
        {workspaceRoles}
      />
    {/if}
  </div>
</div>
