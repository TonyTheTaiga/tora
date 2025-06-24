<script lang="ts">
  import {
    openCreateWorkspaceModal,
    getCreateWorkspaceModal,
  } from "$lib/state/app.svelte";
  import CreateWorkspaceModal from "./create-workspace-modal.svelte";
  import WorkspaceRoleBadge from "$lib/components/workspace-role-badge.svelte";
  import RecentActivity from "$lib/components/recent-activity.svelte";
  import { onMount } from "svelte";

  let { data } = $props();
  let { workspaces } = $derived(data);
  let searchQuery = $state("");

  let filteredWorkspaces = $derived(
    workspaces.filter((workspace) => {
      const query = searchQuery.toLowerCase();
      const searchableText =
        `${workspace.name} ${workspace.description || ""}`.toLowerCase();

      return query.split(" ").every((term) => searchableText.includes(term));
    }),
  );

  let createWorkspaceModal = $derived(getCreateWorkspaceModal());

  const handleKeydown = (event: KeyboardEvent) => {
    if (event.key === "/") {
      event.preventDefault();
      const searchElement = document.querySelector<HTMLInputElement>(
        'input[type="search"]',
      );
      searchElement?.focus();
    }
  };

  onMount(() => {
    window.addEventListener("keydown", handleKeydown);

    return () => {
      window.removeEventListener("keydown", handleKeydown);
    };
  });
</script>

{#if createWorkspaceModal}
  <CreateWorkspaceModal />
{/if}

<div class="font-mono">
  <!-- Header -->
  <div
    class="flex items-center justify-between p-4 md:p-6 border-b border-ctp-surface0/10"
  >
    <div
      class="flex items-stretch gap-3 md:gap-4 min-w-0 flex-1 pr-4 min-h-fit"
    >
      <div
        class="w-2 bg-ctp-blue rounded-full flex-shrink-0 self-stretch"
      ></div>
      <div class="min-w-0 flex-1 py-1">
        <h1 class="text-lg md:text-xl text-ctp-text truncate font-mono">
          Workspaces
        </h1>
        <div class="text-sm text-ctp-subtext0 space-y-1">
          <div>
            {workspaces.length} workspace{workspaces.length !== 1 ? "s" : ""}
          </div>
        </div>
      </div>
    </div>

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
  </div>

  <!-- Search and filter bar -->
  <div class="px-4 md:px-6 py-4">
    <div class="max-w-lg">
      <div
        class="flex items-center bg-ctp-surface0/20 focus-within:ring-1 focus-within:ring-ctp-text/20 transition-all"
      >
        <span class="text-ctp-subtext0 font-mono text-sm px-4 py-3">/</span>
        <input
          id="workspace-search"
          type="search"
          placeholder="search workspaces..."
          bind:value={searchQuery}
          class="flex-1 bg-transparent border-0 py-3 pr-4 text-ctp-text placeholder-ctp-subtext0 focus:outline-none font-mono text-sm"
        />
      </div>
    </div>
  </div>

  <!-- Terminal-style workspace display -->
  <div class="px-4 md:px-6 font-mono">
    {#if filteredWorkspaces.length === 0 && searchQuery}
      <div class="text-ctp-subtext0 text-base">
        <div>search "{searchQuery}"</div>
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
