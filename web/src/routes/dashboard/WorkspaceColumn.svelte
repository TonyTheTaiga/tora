<script lang="ts">
  import {
    openCreateWorkspaceModal,
    getCreateWorkspaceModal,
  } from "$lib/state/modal.svelte";
  import { setSelectedWorkspace } from "./state.svelte";
  import WorkspaceList from "$lib/components/lists/WorkspaceList.svelte";
  import CreateWorkspaceModal from "$lib/components/modals/create-workspace-modal.svelte";

  let { workspaces, workspaceRoles, workspaceInvitations } = $props();
  let workspaceSearchQuery = $state("");
  let createWorkspaceModal = $derived(getCreateWorkspaceModal());
</script>

{#if createWorkspaceModal}
  <CreateWorkspaceModal />
{/if}

<div class="terminal-chrome-header">
  <div class="flex items-center justify-between mb-3">
    <h2 class="text-ctp-text font-medium text-base">workspaces</h2>
    <div>
      <button
        aria-label="create-workspace"
        onclick={() => openCreateWorkspaceModal()}
        class="floating-element px-3 py-2 md:px-4 flex-shrink-0"
      >
        <div class="flex items-center gap-2">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
            class="lucide lucide-folder-plus-icon lucide-folder-plus"
            ><path d="M12 10v6" /><path d="M9 13h6" /><path
              d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"
            /></svg
          >
        </div>
      </button>

      <button
        aria-label="pending-invitations"
        class="floating-element px-3 py-2 md:px-4 flex-shrink-0"
      >
        <div>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
            class="lucide lucide-mail-icon lucide-mail"
            ><path d="m22 7-8.991 5.727a2 2 0 0 1-2.009 0L2 7" /><rect
              x="2"
              y="4"
              width="20"
              height="16"
              rx="2"
            /></svg
          >
        </div>
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
