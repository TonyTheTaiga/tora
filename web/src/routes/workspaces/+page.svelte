<script lang="ts">
  import {
    openCreateWorkspaceModal,
    getCreateWorkspaceModal,
  } from "$lib/state/app.svelte.js";
  import { CreateWorkspaceModal } from "$lib/components/modals";
  import { PageHeader, SearchInput } from "$lib/components";
  import { WorkspaceList } from "$lib/components/lists";
  import RecentActivity from "$lib/components/recent-activity.svelte";

  let { data } = $props();
  let { workspaces } = $derived(data);

  let recentExperiments = $derived(data.recentExperiments || []);
  let recentWorkspaces = $derived(data.recentWorkspaces);
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
        class="group relative bg-ctp-surface0/20 backdrop-blur-md border border-ctp-surface0/30 text-ctp-text hover:bg-ctp-surface0/30 hover:border-ctp-surface0/50 px-3 py-2 md:px-4 text-sm font-mono transition-all flex-shrink-0"
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
  <div class="px-4 md:px-6 font-mono">
    <WorkspaceList {workspaces} {searchQuery} />

    <!-- Recent activity section -->
    <div class="mt-8 border-t border-ctp-surface0/20 pt-6">
      <div class="flex items-center gap-2 mb-3">
        <div class="text-base text-ctp-text font-mono">recent activity</div>
      </div>
      <div class="subtle-surface p-4 text-sm">
        <RecentActivity
          experiments={recentExperiments}
          workspaces={recentWorkspaces}
        />
      </div>
    </div>
  </div>
</div>
