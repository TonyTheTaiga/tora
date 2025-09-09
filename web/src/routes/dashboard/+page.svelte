<script lang="ts">
  import WorkspaceColumn from "./WorkspaceColumn.svelte";
  import ExperimentListColumn from "./ExperimentListColumn.svelte";
  import ExperimentDetails from "./ExperimentDetails.svelte";
  import EmptyState from "./EmptyState.svelte";
  import type {
    ApiResponse,
    WorkspaceRole,
    PendingInvitation,
  } from "$lib/types";
  import { getSelectedWorkspace, getSelectedExperiment } from "./state.svelte";
  import { onMount } from "svelte";
  import { PanelLeftOpen } from "@lucide/svelte";

  let { data } = $props();
  let workspaces = $derived(data.workspaces);
  let selectedWorkspace = $derived(getSelectedWorkspace());
  let selectedExperiment = $derived(getSelectedExperiment());
  let workspaceRoles = $state<WorkspaceRole[]>([]);
  let workspaceInvitations = $state<PendingInvitation[]>([]);

  let navCollapsed = $state(false);
  function openNav() {
    navCollapsed = false;
  }

  function collapseNav() {
    navCollapsed = true;
  }

  const navExpanded = "25%";
  const navCollapsedWidth = "3rem";
  const navTrack = $derived(navCollapsed ? navCollapsedWidth : navExpanded);

  async function loadWorkspaceRoles() {
    try {
      const response = await fetch("/api/workspace-roles");
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const apiReponse: ApiResponse<WorkspaceRole[]> = await response.json();
      workspaceRoles = apiReponse.data.filter((item) => {
        return item.name !== "OWNER";
      });
    } catch (error) {
      console.error(`Failed to load workspace roles:`, error);
    }
  }

  async function loadPendingInvitations() {
    try {
      const response = await fetch("/api/workspace-invitations");
      const responseJson: ApiResponse<PendingInvitation[]> =
        await response.json();
      workspaceInvitations = responseJson.data;
    } catch (error) {
      console.error("Failed to load pending invitations:", error);
    }
  }

  onMount(async () => {
    await loadWorkspaceRoles();
    await loadPendingInvitations();
  });
</script>

<div
  class="min-h-0 min-w-0 grow grid overflow-hidden p-4 gap-2"
  style={`grid-template-columns: ${navTrack} 1fr; transition: grid-template-columns 200ms ease;`}
>
  <!-- Navigation Column: Workspaces -> Experiments, collapses after picking an experiment -->
  <section
    class={`${navCollapsed ? "bg-ctp-surface0/35" : "bg-ctp-surface0/12"} relative shadow-lg shadow-ctp-crust/20 backdrop-blur-sm min-h-0 min-w-0 overflow-y-auto overflow-x-hidden`}
    role="group"
    aria-label="Navigation column"
  >
    {#if navCollapsed}
      <!-- Collapsed rail; expand control lives here -->
      <div
        class="h-full w-full select-none flex items-start justify-center p-2"
      >
        <button
          class="floating-element p-2 rounded-md"
          onclick={openNav}
          title="show navigator"
          aria-label="show navigator"
        >
          <PanelLeftOpen size={16} />
        </button>
      </div>
    {:else if !selectedWorkspace}
      <WorkspaceColumn
        {workspaces}
        {workspaceRoles}
        {workspaceInvitations}
        {collapseNav}
      />
    {:else}
      {#key selectedWorkspace.id}
        <ExperimentListColumn workspace={selectedWorkspace} {collapseNav} />
      {/key}
    {/if}
  </section>

  <!-- Experiment Details Column -->
  <section
    class="bg-ctp-surface0/18 shadow-lg shadow-ctp-crust/20 backdrop-blur-sm min-h-0 min-w-0 overflow-y-auto overflow-x-hidden"
  >
    {#if selectedExperiment}
      {#key selectedExperiment.id}
        <ExperimentDetails experiment={selectedExperiment} />
      {/key}
    {:else if selectedWorkspace}
      <div class="h-full flex items-center justify-center">
        <EmptyState message="select an experiment to view details" />
      </div>
    {:else}
      <div class="h-full flex items-center justify-center">
        <EmptyState message="select a workspace to get started" />
      </div>
    {/if}
  </section>
</div>
