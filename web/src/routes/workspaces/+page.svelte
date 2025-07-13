<script lang="ts">
  import {
    openCreateWorkspaceModal,
    getCreateWorkspaceModal,
  } from "$lib/state/app.svelte.js";
  import { CreateWorkspaceModal } from "$lib/components/modals";
  import { PageHeader, SearchInput } from "$lib/components";
  import { WorkspaceList } from "$lib/components/lists";
  import RecentActivity from "$lib/components/recent-activity.svelte";
  import PendingInvitations from "$lib/components/pending-invitations.svelte";

  let { data } = $props();
  let { workspaces } = $derived(data);

  let recentExperiments = $derived(data.recentExperiments || []);
  let recentWorkspaces = $derived(data.recentWorkspaces);
  let invitations = $derived(data.invitations || []);
  let workspaceRoles = $derived(data.workspaceRoles || []);
  let searchQuery = $state("");

  let createWorkspaceModal = $derived(getCreateWorkspaceModal());
</script>

{#if createWorkspaceModal}
  <CreateWorkspaceModal />
{/if}

<div class="font-mono">
  <!-- Header -->
  <PageHeader
    title="Workspaces"
    subtitle="{workspaces.length} workspace{workspaces.length !== 1 ? 's' : ''}"
  >
    {#snippet actionButton()}
      <button
        onclick={() => openCreateWorkspaceModal()}
        class="floating-element text-ctp-text px-3 py-2 md:px-4 text-sm font-mono flex-shrink-0"
      >
        <div class="flex items-center gap-2">
          <svg
            class="w-4 h-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M12 4v16m8-8H4"
            ></path>
          </svg>
          <span class="hidden sm:inline">New</span>
        </div>
      </button>
    {/snippet}
  </PageHeader>

  <!-- Search and filter bar -->
  <SearchInput
    bind:value={searchQuery}
    placeholder="search workspaces..."
    id="workspace-search"
  />

  <!-- Terminal-style workspace display -->
  <div class="font-mono">
    <WorkspaceList {workspaces} {searchQuery} {workspaceRoles} />

    <!-- Pending invitations section -->
    <PendingInvitations {invitations} />

    <!-- Recent activity section -->
    <div class="section-divider" data-label="recent activity"></div>
    <div class="surface-accent-lavender layer-spacing-md stack-layer">
      <RecentActivity
        experiments={recentExperiments}
        workspaces={recentWorkspaces}
      />
    </div>
  </div>
</div>
