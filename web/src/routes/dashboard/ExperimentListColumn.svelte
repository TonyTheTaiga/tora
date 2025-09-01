<script lang="ts">
  import { getExperimentToEdit } from "$lib/state/modal.svelte";
  import ExperimentList from "$lib/components/lists/ExperimentList.svelte";
  import EditExperimentModal from "$lib/components/modals/edit-experiment-modal.svelte";
  import type { Workspace, Experiment } from "$lib/types";
  import { copyToClipboard } from "$lib/utils/common";
  import {
    setSelectedExperiment,
    loading,
    errors,
    getCachedExperiments,
    setCachedExperiments,
    isWorkspaceLoaded,
  } from "./state.svelte";

  let { workspace }: { workspace: Workspace } = $props();
  let experimentSearchQuery = $state("");
  let experiments: Experiment[] = $state([]);
  let experimentToEdit = $derived(getExperimentToEdit());

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

  $effect(() => {
    const cached = getCachedExperiments(workspace.id);
    if (cached) {
      experiments = cached;
      return;
    }

    if (!isWorkspaceLoaded(workspace.id)) {
      loadExperiments(workspace).then((results) => {
        if (results) {
          experiments = results;
          setCachedExperiments(workspace.id, results);
        }
      });
    }
  });
</script>

{#if experimentToEdit}
  <EditExperimentModal bind:experiment={experimentToEdit} />
{/if}

<div class="flex flex-col">
  <div
    class="sticky top-0 z-10 surface-elevated border-b border-ctp-surface0/30 p-4 min-w-0"
  >
    <div class="flex items-center justify-between mb-3">
      <h2 class="text-ctp-text font-medium text-base">Experiments</h2>
    </div>
    <div class="mb-3">
      <button
        class="text-xs text-ctp-overlay0 hover:text-ctp-blue cursor-pointer"
        tabindex="0"
        onclick={(e) => {
          e.stopPropagation();
          copyToClipboard(workspace.id);
        }}
        title="click to copy workspace id"
      >
        workspace id: {workspace.id}
      </button>
    </div>
    <div
      class="flex items-center bg-ctp-surface0/30 focus-within:ring-1 focus-within:ring-ctp-blue/30 transition-all mb-3 border border-ctp-surface0/40 min-w-0 overflow-hidden"
    >
      <input
        type="search"
        bind:value={experimentSearchQuery}
        placeholder="search experiments..."
        class="flex-1 w-full min-w-0 bg-transparent border-0 py-2 px-3 text-ctp-text placeholder-ctp-subtext0 focus:outline-none text-sm truncate"
      />
    </div>
  </div>

  <!-- Content Area -->
  <div class="p-4">
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
        {workspace}
        searchQuery={experimentSearchQuery}
        onItemClick={setSelectedExperiment}
      />
    {/if}
  </div>
</div>
