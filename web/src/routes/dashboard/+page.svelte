<script lang="ts">
  import { getCreateWorkspaceModal } from "$lib/state/app.svelte";
  import {
    getSelectedWorkspace,
    getSelectedExperiment,
    setSelectedExperiment,
    loading,
    errors,
  } from "./state.svelte";
  import CreateWorkspaceModal from "$lib/components/modals/create-workspace-modal.svelte";
  import type {
    Experiment,
    PendingInvitation,
    ApiResponse,
    WorkspaceRole,
  } from "$lib/types";
  import { onMount } from "svelte";
  import WorkspaceColumn from "./WorkspaceColumn.svelte";
  import ExperimentListColumn from "./ExperimentListColumn.svelte";
  import ExperimentDetails from "./ExperimentDetails.svelte";

  let { data } = $props();
  let workspaces = $derived(data.workspaces);
  let workspaceRoles = $state<WorkspaceRole[]>([]);
  let workspaceInvitations = $state<PendingInvitation[]>([]);
  let createWorkspaceModal = $derived(getCreateWorkspaceModal());
  let selectedWorkspace = $derived(getSelectedWorkspace());
  let selectedExperiment = $derived(getSelectedExperiment());
  let experiments = $state<Experiment[]>([]);

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
      console.error("Failed to load pending invitaitons");
    }
  }

  async function loadExperiments(workspaceId: string) {
    try {
      loading.experiments = true;
      errors.experiments = null;
      const response = await fetch(
        `/api/workspaces/${workspaceId}/experiments`,
      );
      if (!response.ok)
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      const apiResponse = await response.json();
      const data = apiResponse.data;
      if (!data || !Array.isArray(data))
        throw new Error("Invalid response structure from experiments API");

      experiments = data.map((exp: any) => ({
        id: exp.id,
        name: exp.name,
        description: exp.description || "",
        hyperparams: exp.hyperparams || [],
        tags: exp.tags || [],
        createdAt: new Date(exp.created_at),
        updatedAt: new Date(exp.updated_at),
        availableMetrics: exp.available_metrics || [],
        workspaceId: workspaceId,
      }));
    } catch (error) {
      console.error("Failed to load experiments:", error);
      errors.experiments =
        error instanceof Error ? error.message : "Failed to load experiments";
    } finally {
      loading.experiments = false;
    }
  }

  onMount(async () => {
    await loadWorkspaceRoles();
    await loadPendingInvitations();
  });

  $effect(() => {
    if (selectedWorkspace) {
      setSelectedExperiment(null);
      loadExperiments(selectedWorkspace.id);
    }
  });
</script>

{#if createWorkspaceModal}
  <CreateWorkspaceModal />
{/if}

<div
  class="bg-ctp-base text-ctp-text flex space-x-2 font-mono border-ctp-surface0/30"
>
  <div class="w-1/4 border-r border-b border-ctp-surface0/30 flex flex-col">
    <WorkspaceColumn {workspaces} {workspaceRoles} />
  </div>

  <div
    class="w-1/4 border-r border-l border-b border-ctp-surface0/30 flex flex-col"
  >
    {#if selectedWorkspace}
      <ExperimentListColumn {experiments} />
    {:else}
      <div class="text-ctp-subtext0 text-sm terminal-chrome-header">
        select a workspace to view experiments
      </div>
    {/if}
  </div>

  <div class="w-1/2 border-l border-b border-ctp-surface0/30 flex flex-col">
    {#if selectedExperiment}
      <ExperimentDetails {selectedExperiment} />
    {:else}
      <div class="text-ctp-subtext0 text-sm terminal-chrome-header">
        select a experiemnt to view details
      </div>
    {/if}
  </div>
</div>
