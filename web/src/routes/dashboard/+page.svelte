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

  let { data } = $props();
  let workspaces = $derived(data.workspaces);
  let selectedWorkspace = $derived(getSelectedWorkspace());
  let selectedExperiment = $derived(getSelectedExperiment());
  let workspaceRoles = $state<WorkspaceRole[]>([]);
  let workspaceInvitations = $state<PendingInvitation[]>([]);

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
  class="text-ctp-text flex space-x-2 font-mono border-ctp-surface0/30 p-4 h-[calc(100vh-4.5rem)] overflow-hidden"
>
  <div
    class="w-1/4 border-l border-r border-b border-ctp-surface0/30 flex flex-col overflow-y-auto"
  >
    <WorkspaceColumn {workspaces} {workspaceRoles} {workspaceInvitations} />
  </div>

  <div
    class="w-1/4 border-r border-l border-b border-ctp-surface0/30 flex flex-col overflow-y-auto"
  >
    {#if selectedWorkspace}
      <ExperimentListColumn workspace={selectedWorkspace} />
    {:else}
      <EmptyState message="select a workspace to view experiments" />
    {/if}
  </div>

  <div
    class="w-1/2 border-l border-r border-b border-ctp-surface0/30 flex flex-col overflow-y-auto"
  >
    {#if selectedExperiment}
      <ExperimentDetails experiment={selectedExperiment} />
    {:else}
      <EmptyState message="select a experiment to view details" />
    {/if}
  </div>
</div>
