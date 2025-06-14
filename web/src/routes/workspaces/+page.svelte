<script lang="ts">
  import { openCreateWorkspaceModal, getCreateWorkspaceModal } from "$lib/state/app.svelte";
  import CreateWorkspaceModal from "./create-workspace-modal.svelte";
  import WorkspaceRoleBadge from "$lib/components/workspace-role-badge.svelte";
  
  let { data } = $props();
  let { workspaces } = $derived(data);
  let searchQuery = $state("");
  
  let filteredWorkspaces = $derived(
    workspaces.filter(workspace => 
      workspace.name.toLowerCase().includes(searchQuery.toLowerCase())
    )
  );
  
  let createWorkspaceModal = $derived(getCreateWorkspaceModal());
</script>

{#if createWorkspaceModal}
  <CreateWorkspaceModal />
{/if}

<div class="flex-1 p-2 sm:p-4 max-w-none mx-2 sm:mx-4">
  <div class="max-w-6xl mx-auto space-y-6">
  <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
    <div>
      <h1 class="text-3xl font-bold text-ctp-text">Workspaces</h1>
      <p class="text-ctp-subtext1 mt-1">Manage and access your workspaces</p>
    </div>
    
    <button 
      onclick={() => openCreateWorkspaceModal()}
      class="inline-flex items-center justify-center gap-2 px-6 py-3 font-medium rounded-full bg-ctp-blue/20 border border-ctp-blue/40 text-ctp-blue hover:bg-ctp-blue hover:text-ctp-crust transition-all duration-200 backdrop-blur-sm"
    >
      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"></path>
      </svg>
      Create Workspace
    </button>
  </div>

  <div class="relative">
    <input
      type="text"
      placeholder="Search workspaces..."
      bind:value={searchQuery}
      class="w-full bg-ctp-surface0/30 backdrop-blur-sm border border-ctp-surface0/40 rounded-lg px-4 py-3 pl-10 text-ctp-text placeholder-ctp-subtext0 focus:outline-none focus:ring-2 focus:ring-ctp-blue/50 focus:border-ctp-blue/50 transition-all"
    />
    <svg class="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-ctp-subtext0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
    </svg>
  </div>

  {#if filteredWorkspaces.length === 0 && searchQuery}
    <div class="text-center py-12">
      <div class="text-ctp-subtext0 text-lg">No workspaces found matching "{searchQuery}"</div>
    </div>
  {:else if workspaces.length === 0}
    <div class="text-center py-16">
      <div class="bg-ctp-surface0/10 backdrop-blur-md rounded-2xl border border-ctp-surface0/20 p-8 max-w-md mx-auto shadow-xl">
        <div class="text-ctp-subtext0 text-lg mb-4">No workspaces yet</div>
        <p class="text-ctp-subtext1 mb-6">Create your first workspace to get started with experiments and data analysis.</p>
        <button 
          onclick={() => openCreateWorkspaceModal()}
          class="inline-flex items-center justify-center gap-2 px-6 py-3 font-medium rounded-full bg-ctp-blue/20 border border-ctp-blue/40 text-ctp-blue hover:bg-ctp-blue hover:text-ctp-crust transition-all duration-200 backdrop-blur-sm"
        >
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"></path>
          </svg>
          Create Your First Workspace
        </button>
      </div>
    </div>
  {:else}
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {#each filteredWorkspaces as workspace}
        <a 
          href={`/workspaces/${workspace.id}`}
          class="group bg-ctp-surface0/10 backdrop-blur-md hover:bg-ctp-surface0/20 border border-ctp-surface0/20 hover:border-ctp-surface0/40 rounded-2xl p-6 transition-all duration-200 hover:scale-105 shadow-xl"
        >
          <div class="flex items-start justify-between mb-4">
            <div class="bg-ctp-blue/10 p-3 rounded-lg">
              <svg class="w-6 h-6 text-ctp-blue" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4"></path>
              </svg>
            </div>
            <div class="opacity-0 group-hover:opacity-100 transition-opacity">
              <svg class="w-4 h-4 text-ctp-subtext0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path>
              </svg>
            </div>
          </div>
          
          <h3 class="text-xl font-semibold text-ctp-text mb-2 group-hover:text-ctp-blue transition-colors">
            {workspace.name}
          </h3>
          
          <p class="text-ctp-subtext1 text-sm mb-4 line-clamp-2">
            {workspace.description || "No description available"}
          </p>
          
          <div class="flex items-center justify-between text-sm">
            <div class="flex items-center space-x-4">
              <span class="text-ctp-subtext0">
                <svg class="w-4 h-4 inline mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197m13.5-9a2.5 2.5 0 11-5 0 2.5 2.5 0 015 0z"></path>
                </svg>
                1 member
              </span>
            </div>
            
            <WorkspaceRoleBadge role={workspace.role || "VIEWER"} />
          </div>
        </a>
      {/each}
    </div>
  {/if}
  </div>
</div>
