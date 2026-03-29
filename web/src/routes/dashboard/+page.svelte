<script lang="ts">
  import ExperimentDetails from "./ExperimentDetails.svelte";
  import EmptyState from "./EmptyState.svelte";
  import NavBottomBreadcrumb from "./NavBottomBreadcrumb.svelte";
  import type { ApiResponse, PendingInvitation } from "$lib/types";
  import { getSelectedWorkspace, getSelectedExperiment } from "./state.svelte";
  import { onMount } from "svelte";

  let { data } = $props();
  let workspaces = $derived(data.workspaces);
  let selectedWorkspace = $derived(getSelectedWorkspace());
  let selectedExperiment = $derived(getSelectedExperiment());
  let workspaceInvitations = $state<PendingInvitation[]>([]);

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
    await loadPendingInvitations();
  });
</script>

<div class="min-h-0 min-w-0 grow flex flex-col overflow-hidden">
  <section
    class="flex-1 bg-ctp-surface0/18 shadow-lg shadow-ctp-crust/20 backdrop-blur-sm min-h-0 overflow-y-auto overflow-x-hidden m-4 mb-2"
  >
    {#if selectedExperiment}
      {#key selectedExperiment.id}
        <ExperimentDetails experiment={selectedExperiment} />
      {/key}
    {:else}
      <div class="h-full flex items-center justify-center">
        <EmptyState
          message={selectedWorkspace
            ? "select an experiment from the breadcrumb below"
            : "select a workspace from the breadcrumb below"}
        />
      </div>
    {/if}
  </section>
  <NavBottomBreadcrumb {workspaces} />
</div>
