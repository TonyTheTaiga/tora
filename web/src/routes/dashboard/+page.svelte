<script lang="ts">
  import { getSelectedWorkspace, getSelectedExperiment } from "./state.svelte";
  import WorkspaceColumn from "./WorkspaceColumn.svelte";
  import ExperimentListColumn from "./ExperimentListColumn.svelte";
  import ExperimentDetails from "./ExperimentDetails.svelte";
  import EmptyState from "./EmptyState.svelte";
  import { loadData } from "./loader.svelte";

  let { data } = $props();
  let workspaces = $derived(data.workspaces);
  let selectedWorkspace = $derived(getSelectedWorkspace());
  let selectedExperiment = $derived(getSelectedExperiment());
  const { workspaceRoles, workspaceInvitations } = loadData();
</script>

<div
  class="bg-ctp-base text-ctp-text flex space-x-2 font-mono border-ctp-surface0/30"
>
  <div class="w-1/4 border-r border-b border-ctp-surface0/30 flex flex-col">
    <WorkspaceColumn {workspaces} {workspaceRoles} {workspaceInvitations} />
  </div>

  <div
    class="w-1/4 border-r border-l border-b border-ctp-surface0/30 flex flex-col"
  >
    {#if selectedWorkspace}
      <ExperimentListColumn workspace={selectedWorkspace} />
    {:else}
      <EmptyState message="select a workspace to view experiments" />
    {/if}
  </div>

  <div class="w-1/2 border-l border-b border-ctp-surface0/30 flex flex-col">
    {#if selectedExperiment}
      <ExperimentDetails experiment={selectedExperiment} />
    {:else}
      <EmptyState message="select a experiment to view details" />
    {/if}
  </div>
</div>
