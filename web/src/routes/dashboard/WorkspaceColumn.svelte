<script lang="ts">
  import {
    openCreateWorkspaceModal,
    getCreateWorkspaceModal,
  } from "$lib/state/modal.svelte";
  import { setSelectedWorkspace } from "./state.svelte";
  import WorkspaceList from "$lib/components/lists/WorkspaceList.svelte";
  import CreateWorkspaceModal from "$lib/components/modals/create-workspace-modal.svelte";
  import { Mail, FolderPlus } from "@lucide/svelte";

  let { workspaces, workspaceRoles, workspaceInvitations } = $props();
  let workspaceSearchQuery = $state("");
  let createWorkspaceModal = $derived(getCreateWorkspaceModal());
</script>

{#if createWorkspaceModal}
  <CreateWorkspaceModal />
{/if}

<div class="terminal-chrome-header">
  <div class="flex items-center justify-between mb-3">
    <h2 class="text-ctp-text font-medium text-base">Workspaces</h2>
    <div>
      <button
        aria-label="create-workspace"
        onclick={() => openCreateWorkspaceModal()}
        class="floating-element p-2"
      >
        <FolderPlus />
      </button>

      <button aria-label="pending-invitations" class="floating-element p-2">
        <Mail />
      </button>
    </div>
  </div>

  <div
    class="flex items-center bg-ctp-surface0/20 focus-within:ring-1 focus-within:ring-ctp-text/20 transition-all"
  >
    <input
      type="search"
      bind:value={workspaceSearchQuery}
      placeholder="search workspaces..."
      class="flex-1 bg-transparent border-0 py-2 pr-3 text-ctp-text placeholder-ctp-subtext0 focus:outline-none text-sm"
    />
  </div>
</div>

<div class="flex-1 overflow-y-auto min-h-0">
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
