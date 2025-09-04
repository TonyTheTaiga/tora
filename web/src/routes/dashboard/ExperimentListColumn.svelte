<script lang="ts">
  import { getExperimentToEdit } from "$lib/state/modal.svelte";
  import ExperimentList from "$lib/components/lists/ExperimentList.svelte";
  import EditExperimentModal from "$lib/components/modals/edit-experiment-modal.svelte";
  import type { Workspace, Experiment } from "$lib/types";
  import { onMount } from "svelte";
  import { copyToClipboard } from "$lib/utils/common";
  import { RefreshCw } from "@lucide/svelte";
  import {
    setSelectedExperiment,
    loading,
    errors,
    getCachedExperiments,
    setCachedExperiments,
    isWorkspaceLoaded,
  } from "./state.svelte";
  import {
    loadExperimentsFromStorage,
    saveExperimentsToStorage,
    getExperimentsTimestamp,
  } from "$lib/utils/persistentCache";

  let { workspace }: { workspace: Workspace } = $props();
  let experimentSearchQuery = $state("");
  let experiments: Experiment[] = $state([]);
  let experimentToEdit = $derived(getExperimentToEdit());

  function handleExperimentsChange(updated: Experiment[]) {
    experiments = updated;
    setCachedExperiments(workspace.id, updated);
    saveExperimentsToStorage(workspace.id, updated);
  }

  type LoadOpts = { signal?: AbortSignal; silent?: boolean };

  async function loadExperiments(
    workspace: Workspace,
    opts: LoadOpts = {},
  ): Promise<Experiment[] | undefined> {
    try {
      if (!opts.silent) loading.experiments = true;
      errors.experiments = null;
      const response = await fetch(
        `/api/workspaces/${workspace.id}/experiments`,
        { signal: opts.signal },
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
      if ((error as any)?.name === "AbortError") return; // ignore aborts
      console.error("Failed to load experiments:", error);
      errors.experiments =
        error instanceof Error ? error.message : "Failed to load experiments";
    } finally {
      if (!opts.silent) loading.experiments = false;
    }
  }

  let abortController: AbortController | null = null;

  const REVALIDATE_AGE_MS = 2 * 60 * 1000; // 2 minutes

  onMount(() => {
    let hadCached = false;
    const cached = getCachedExperiments(workspace.id);
    if (cached) {
      experiments = cached;
      hadCached = true;
    } else {
      const persisted = loadExperimentsFromStorage(workspace.id);
      if (persisted) {
        experiments = persisted;
        setCachedExperiments(workspace.id, persisted);
        hadCached = true;
      }
    }

    if (!isWorkspaceLoaded(workspace.id)) {
      abortController?.abort();
      abortController = new AbortController();
      loadExperiments(workspace, { signal: abortController.signal }).then(
        (results) => {
          if (results) {
            experiments = results;
            setCachedExperiments(workspace.id, results);
            saveExperimentsToStorage(workspace.id, results);
          }
        },
      );
    }

    // SWR background revalidation based on stored timestamp age
    const ts = getExperimentsTimestamp(workspace.id);
    const age = ts ? Date.now() - ts : Number.POSITIVE_INFINITY;
    if (hadCached && age > REVALIDATE_AGE_MS) {
      const swrController = new AbortController();
      loadExperiments(workspace, {
        signal: swrController.signal,
        silent: true,
      }).then((results) => {
        if (results) {
          experiments = results;
          setCachedExperiments(workspace.id, results);
          saveExperimentsToStorage(workspace.id, results);
        }
      });
    }

    return () => {
      abortController?.abort();
    };
  });

  async function refreshExperiments() {
    try {
      errors.experiments = null;
      loading.experiments = true;
      abortController?.abort();
      abortController = new AbortController();
      const results = await loadExperiments(workspace, {
        signal: abortController.signal,
        silent: false,
      });
      if (results) {
        experiments = results;
        setCachedExperiments(workspace.id, results);
        saveExperimentsToStorage(workspace.id, results);
      }
    } finally {
      loading.experiments = false;
    }
  }
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
      <button
        class="inline-flex items-center gap-1 text-xs bg-transparent border border-ctp-surface0/40 hover:border-ctp-surface0/60 text-ctp-overlay0 hover:text-ctp-text px-2 py-1 transition-colors disabled:opacity-50"
        onclick={refreshExperiments}
        disabled={loading.experiments}
        title="refresh experiments"
      >
        <RefreshCw class="w-3.5 h-3.5" />
        refresh
      </button>
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
        onExperimentsChange={handleExperimentsChange}
        onItemClick={setSelectedExperiment}
      />
    {/if}
  </div>
</div>
