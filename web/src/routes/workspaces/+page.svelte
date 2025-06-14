<script lang="ts">
  import {
    openCreateWorkspaceModal,
    getCreateWorkspaceModal,
  } from "$lib/state/app.svelte";
  import CreateWorkspaceModal from "./create-workspace-modal.svelte";
  import WorkspaceRoleBadge from "$lib/components/workspace-role-badge.svelte";
  import RecentActivity from "$lib/components/recent-activity.svelte";

  let { data } = $props();
  let { workspaces } = $derived(data);
  let searchQuery = $state("");

  let filteredWorkspaces = $derived(
    workspaces.filter((workspace) =>
      workspace.name.toLowerCase().includes(searchQuery.toLowerCase()),
    ),
  );

  let createWorkspaceModal = $derived(getCreateWorkspaceModal());
</script>

{#if createWorkspaceModal}
  <CreateWorkspaceModal />
{/if}

<div class="bg-gradient-to-br from-ctp-base via-ctp-base to-ctp-mantle">
  <!-- Minimal top bar -->
  <div class="flex items-center justify-between p-6">
    <div class="flex items-center gap-4">
      <div class="w-2 h-8 bg-ctp-blue rounded-full"></div>
      <div>
        <h1 class="text-xl font-bold text-ctp-text">Workspaces</h1>
        <div class="text-xs text-ctp-subtext0 font-mono">
          {workspaces.length} total
        </div>
      </div>
    </div>

    <button
      onclick={() => openCreateWorkspaceModal()}
      class="group relative bg-ctp-surface0/20 backdrop-blur-md border border-ctp-surface0/30 text-ctp-text hover:bg-ctp-surface0/30 hover:border-ctp-surface0/50 rounded-full px-4 py-2 text-sm font-medium transition-all duration-300 hover:scale-105 active:scale-95"
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
        <span>New</span>
      </div>
    </button>
  </div>

  <!-- Search and filter bar -->
  <div class="px-6 pb-8">
    <div class="max-w-lg">
      <div class="relative">
        <input
          type="text"
          placeholder="Search or filter workspaces..."
          bind:value={searchQuery}
          class="w-full bg-ctp-surface0/20 border-0 px-4 py-3 text-ctp-text placeholder-ctp-subtext0 focus:outline-none focus:ring-1 focus:ring-ctp-text/20 transition-all font-mono text-sm"
        />
        <div
          class="absolute right-3 top-1/2 transform -translate-y-1/2 text-xs text-ctp-subtext0 font-mono"
        >
          {filteredWorkspaces.length}/{workspaces.length}
        </div>
      </div>
    </div>
  </div>

  <!-- Terminal-style workspace display -->
  <div class="px-6 font-mono">
    {#if filteredWorkspaces.length === 0 && searchQuery}
      <div class="text-ctp-subtext0 text-sm">
        <div>$ search "{searchQuery}"</div>
        <div class="text-ctp-subtext1 ml-2">no results found</div>
      </div>
    {:else if workspaces.length === 0}
      <div class="space-y-2 text-sm">
        <div class="text-ctp-subtext0">$ ls -la workspaces/</div>
        <div class="text-ctp-subtext1 ml-2">total 0</div>
        <div class="text-ctp-subtext1 ml-2">directory empty</div>
        <div class="mt-4">
          <button
            onclick={() => openCreateWorkspaceModal()}
            class="text-ctp-blue hover:text-ctp-blue/80 transition-colors"
          >
            $ mkdir new_workspace
          </button>
        </div>
      </div>
    {:else}
      <!-- File listing style layout -->
      <div class="space-y-1">
        <!-- Header -->
        <div
          class="flex items-center text-xs text-ctp-subtext0 pb-2 border-b border-ctp-surface0/20"
        >
          <div class="w-4">•</div>
          <div class="flex-1">name</div>
          <div class="w-16 text-right">role</div>
          <div class="w-20 text-right">modified</div>
          <div class="w-16 text-right">status</div>
        </div>

        <!-- Workspace entries -->
        {#each filteredWorkspaces as workspace}
          <a
            href={`/workspaces/${workspace.id}`}
            class="group flex items-center text-sm hover:bg-ctp-surface0/20 px-1 py-2 transition-colors"
          >
            <div class="w-4 text-ctp-green text-xs">●</div>
            <div class="flex-1 min-w-0">
              <div class="flex items-center gap-2">
                <span
                  class="text-ctp-text group-hover:text-ctp-blue transition-colors font-medium truncate"
                >
                  {workspace.name}
                </span>
                {#if workspace.description}
                  <span class="text-ctp-subtext1 text-xs truncate">
                    - {workspace.description}
                  </span>
                {/if}
              </div>
            </div>
            <div class="w-16 text-right text-xs text-ctp-subtext0">
              <WorkspaceRoleBadge role={workspace.role || "VIEWER"} />
            </div>
            <div class="w-20 text-right text-xs text-ctp-subtext0">
              {new Date(workspace.createdAt).toLocaleDateString("en-US", {
                month: "short",
                day: "numeric",
              })}
            </div>
            <div class="w-16 text-right text-xs text-ctp-green">active</div>
          </a>
        {/each}

        <!-- Summary line -->
        <div
          class="flex items-center text-xs text-ctp-subtext0 pt-2 border-t border-ctp-surface0/20"
        >
          <div class="flex-1">
            {filteredWorkspaces.length} workspace{filteredWorkspaces.length !==
            1
              ? "s"
              : ""} total
          </div>
        </div>
      </div>

      <!-- Recent activity section -->
      <div class="mt-8 border-t border-ctp-surface0/20 pt-6">
        <div class="flex items-center gap-2 mb-3">
          <div class="text-sm text-ctp-text font-mono">recent activity</div>
        </div>
        <div class="bg-ctp-surface0/10 p-4 text-xs">
          <RecentActivity
            experiments={data.recentExperiments}
            workspaces={data.recentWorkspaces}
          />
        </div>
      </div>
    {/if}
  </div>
</div>
