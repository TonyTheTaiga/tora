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

<div class="font-mono">
  <!-- Minimal top bar -->
  <div class="flex items-center justify-between p-6">
    <div class="flex items-stretch gap-4 min-h-fit">
      <div class="w-2 bg-ctp-blue rounded-full self-stretch"></div>
      <div class="py-1">
        <h1 class="text-xl font-bold text-ctp-text">Workspaces</h1>
        <div class="text-sm text-ctp-subtext0 font-mono">
          {workspaces.length} total
        </div>
      </div>
    </div>

    <button
      onclick={() => openCreateWorkspaceModal()}
      class="bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-text hover:bg-ctp-surface0/30 hover:border-ctp-surface0/50 px-4 py-2 text-sm transition-all"
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
          class="absolute right-3 top-1/2 transform -translate-y-1/2 text-sm text-ctp-subtext0 font-mono"
        >
          {filteredWorkspaces.length}/{workspaces.length}
        </div>
      </div>
    </div>
  </div>

  <!-- Terminal-style workspace display -->
  <div class="px-6 font-mono">
    {#if filteredWorkspaces.length === 0 && searchQuery}
      <div class="text-ctp-subtext0 text-base">
        <div>$ search "{searchQuery}"</div>
        <div class="text-ctp-subtext1 ml-2">no results found</div>
      </div>
    {:else}
      <!-- File listing style layout -->
      <div>
        <table class="w-full table-fixed">
          <thead>
            <tr
              class="text-sm text-ctp-subtext0 border-b border-ctp-surface0/20"
            >
              <th class="text-left py-2 w-4">•</th>
              <th class="text-left py-2">name</th>
              <th class="text-right py-2 w-24">role</th>
              <th class="text-right py-2 w-24">modified</th>
            </tr>
          </thead>
          <tbody>
            {#each filteredWorkspaces as workspace}
              <tr
                class="group text-base hover:bg-ctp-surface0/20 transition-colors"
              >
                <td class="py-2 px-1 w-4">
                  <div class="text-ctp-green text-sm"></div>
                </td>
                <td class="py-2 px-1 min-w-0">
                  <a href={`/workspaces/${workspace.id}`} class="block min-w-0">
                    <div class="truncate text-ctp-text">
                      <span
                        class="group-hover:text-ctp-blue transition-colors font-medium"
                      >
                        {workspace.name}
                      </span>
                      {#if workspace.description}
                        <span class="text-ctp-subtext1 text-sm">
                          - {workspace.description}
                        </span>
                      {/if}
                    </div>
                  </a>
                </td>
                <td class="py-2 px-1 text-right text-sm text-ctp-subtext0 w-24">
                  <WorkspaceRoleBadge role={workspace.role || "VIEWER"} />
                </td>
                <td class="py-2 px-1 text-right text-sm text-ctp-subtext0 w-24">
                  {new Date(workspace.createdAt).toLocaleDateString("en-US", {
                    month: "short",
                    day: "numeric",
                  })}
                </td>
              </tr>
            {/each}
          </tbody>
        </table>

        <!-- Summary line -->
        <div
          class="flex items-center text-sm text-ctp-subtext0 pt-2 border-t border-ctp-surface0/20"
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
          <div class="text-base text-ctp-text font-mono">recent activity</div>
        </div>
        <div class="bg-ctp-surface0/10 p-4 text-sm">
          <RecentActivity
            experiments={data.recentExperiments}
            workspaces={data.recentWorkspaces}
          />
        </div>
      </div>
    {/if}
  </div>
</div>
