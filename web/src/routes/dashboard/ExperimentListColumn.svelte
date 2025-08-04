<script lang="ts">
  import ExperimentList from "$lib/components/lists/ExperimentList.svelte";
  import type { Workspace, Experiment } from "$lib/types";
  import { copyToClipboard } from "$lib/utils/common";
  import {
    getSelectedWorkspace,
    setSelectedExperiment,
    loading,
    errors,
  } from "./state.svelte";
  let selectedWorkspace = $derived(getSelectedWorkspace());
  let experimentSearchQuery = $state("");
  let experiments: Experiment[] = $state([]);

  $effect(() => {
    if (selectedWorkspace) {
      loadExperiments(selectedWorkspace).then((results) => {
        if (results) {
          experiments = results;
        }
      });
    }
  });

  async function loadExperiments(
    workspace: Workspace,
  ): Promise<Experiment[] | undefined> {
    try {
      loading.experiments = true;
      errors.experiments = null;
      const response = await fetch(
        `/api/workspaces/${workspace.id}/experiments`,
      );
      if (!response.ok)
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      const apiResponse = await response.json();
      const data = apiResponse.data;
      if (!data || !Array.isArray(data))
        throw new Error("Invalid response structure from experiments API");

      return data.map((exp: any) => ({
        id: exp.id,
        name: exp.name,
        description: exp.description || "",
        hyperparams: exp.hyperparams || [],
        tags: exp.tags || [],
        createdAt: new Date(exp.created_at),
        updatedAt: new Date(exp.updated_at),
        availableMetrics: exp.available_metrics || [],
        workspaceId: workspace.id,
      }));
    } catch (error) {
      console.error("Failed to load experiments:", error);
      errors.experiments =
        error instanceof Error ? error.message : "Failed to load experiments";
    } finally {
      loading.experiments = false;
    }
  }
</script>

<div class="terminal-chrome-header">
  <div class="flex items-center justify-between mb-3">
    <h2 class="text-ctp-text font-medium text-base">experiments</h2>
    <span
      class="bg-ctp-surface0/20 text-ctp-subtext0 px-2 py-1 text-xs border border-ctp-surface0/30"
      >[{experiments ? experiments.length : 0}]</span
    >
  </div>
  <div class="mb-3">
    <span
      class="text-xs text-ctp-overlay0 hover:text-ctp-blue cursor-pointer"
      role="button"
      tabindex="0"
      onclick={(e) => {
        e.stopPropagation();
        selectedWorkspace && copyToClipboard(selectedWorkspace.id);
      }}
      onkeydown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          e.stopPropagation();
          selectedWorkspace && copyToClipboard(selectedWorkspace.id);
        }
      }}
      title="click to copy workspace id"
    >
      workspace id: {selectedWorkspace && selectedWorkspace.id}
    </span>
  </div>
  <div
    class="flex items-center bg-ctp-surface0/20 focus-within:ring-1 focus-within:ring-ctp-text/20"
  >
    <input
      type="search"
      bind:value={experimentSearchQuery}
      placeholder="search experiments..."
      class="flex-1 bg-transparent border-0 py-2 pr-3 text-ctp-text placeholder-ctp-subtext0 focus:outline-none text-sm"
    />
  </div>
</div>

<div class="flex-1 overflow-y-auto min-h-0">
  {#if selectedWorkspace}
    {#if loading.experiments}
      <div class="text-center py-8 text-ctp-subtext0 text-sm">
        loading experiments...
      </div>
    {:else if errors.experiments}
      <div class="surface-layer-2 p-4 m-2">
        <div class="text-ctp-red font-medium mb-2 text-sm">
          error loading experiments
        </div>
        <div class="text-ctp-subtext0 text-xs mb-3">
          {errors.experiments}
        </div>
      </div>
    {:else if experiments && experiments.length === 0}
      <div class="text-center py-8 text-ctp-subtext0 text-sm">
        no experiments found
      </div>
    {:else}
      <ExperimentList
        {experiments}
        searchQuery={experimentSearchQuery}
        onItemClick={setSelectedExperiment}
      />
    {/if}
  {/if}
</div>
