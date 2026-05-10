<script lang="ts">
  import {
    ChevronRight,
    Folder,
    FlaskConical,
    ChevronUp,
  } from "@lucide/svelte";
  import type { Workspace, Experiment } from "$lib/types";
  import {
    getSelectedWorkspace,
    setSelectedWorkspace,
    getSelectedExperiment,
    setSelectedExperiment,
    getCachedExperiments,
    setCachedExperiments,
  } from "./state.svelte";

  let {
    workspaces,
  }: {
    workspaces: Workspace[];
  } = $props();

  let selectedWorkspace = $derived(getSelectedWorkspace());
  let selectedExperiment = $derived(getSelectedExperiment());

  let showWorkspaceDropdown = $state(false);
  let showExperimentDropdown = $state(false);
  let workspaceQuery = $state("");
  let experimentQuery = $state("");

  let experiments = $state<Experiment[]>([]);
  let fetchController: AbortController | null = null;

  $effect(() => {
    fetchController?.abort();
    fetchController = null;

    if (selectedWorkspace) {
      const cached = getCachedExperiments(selectedWorkspace.id);
      if (cached) {
        experiments = cached;
      } else {
        const wsId = selectedWorkspace.id;
        const ac = new AbortController();
        fetchController = ac;
        fetch(`/api/workspaces/${wsId}/experiments`, { signal: ac.signal })
          .then((r) => {
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            return r.json();
          })
          .then((res) => {
            if (ac.signal.aborted) return;
            const data = (res.data || []).map((exp: any) => ({
              id: exp.id,
              name: exp.name,
              description: exp.description || "",
              hyperparams: exp.hyperparams || [],
              tags: exp.tags || [],
              createdAt: new Date(exp.created_at),
              updatedAt: new Date(exp.updated_at),
              workspaceId: wsId,
            }));
            experiments = data;
            setCachedExperiments(wsId, data);
          })
          .catch((e) => {
            if (e?.name === "AbortError") return;
            console.error("Failed to load experiments:", e);
          });
      }
    } else {
      experiments = [];
    }
  });

  let filteredWorkspaces = $derived(
    workspaceQuery
      ? workspaces.filter((w) =>
          w.name.toLowerCase().includes(workspaceQuery.toLowerCase()),
        )
      : workspaces,
  );

  let filteredExperiments = $derived(
    experimentQuery
      ? experiments.filter((e) =>
          e.name.toLowerCase().includes(experimentQuery.toLowerCase()),
        )
      : experiments,
  );

  function selectWorkspace(w: Workspace) {
    setSelectedWorkspace(w);
    setSelectedExperiment(null);
    showWorkspaceDropdown = false;
    workspaceQuery = "";
  }

  function selectExperiment(e: Experiment) {
    setSelectedExperiment(e);
    showExperimentDropdown = false;
    experimentQuery = "";
  }

  function toggleWorkspaceDropdown() {
    showWorkspaceDropdown = !showWorkspaceDropdown;
    showExperimentDropdown = false;
    workspaceQuery = "";
  }

  function toggleExperimentDropdown() {
    showExperimentDropdown = !showExperimentDropdown;
    showWorkspaceDropdown = false;
    experimentQuery = "";
  }

  function closeAll() {
    showWorkspaceDropdown = false;
    showExperimentDropdown = false;
  }
</script>

<!-- Backdrop to close dropdowns -->
{#if showWorkspaceDropdown || showExperimentDropdown}
  <button
    class="fixed inset-0 z-30"
    onclick={closeAll}
    tabindex="-1"
    aria-label="Close dropdown"
  ></button>
{/if}

<!-- Breadcrumb bar -->
<div
  class="sticky bottom-0 z-40 flex items-center gap-1 bg-ctp-mantle/80 backdrop-blur-md border-t border-ctp-surface0/30 px-4 py-2.5 text-sm"
>
  <!-- Workspace segment -->
  <div class="relative">
    <button
      class="flex items-center gap-1.5 px-2 py-1 hover:bg-ctp-surface0/30 transition-colors text-ctp-text"
      onclick={toggleWorkspaceDropdown}
    >
      <Folder size={14} class="text-ctp-overlay0" />
      <span class="max-w-[200px] truncate"
        >{selectedWorkspace?.name ?? "Workspaces"}</span
      >
      <ChevronUp size={12} class="text-ctp-overlay0" />
    </button>

    {#if showWorkspaceDropdown}
      <div
        class="absolute bottom-full left-0 mb-1 w-64 bg-ctp-mantle border border-ctp-surface0/50 shadow-xl shadow-ctp-crust/40 z-50"
      >
        <div class="max-h-[300px] overflow-y-auto py-1">
          {#each filteredWorkspaces as w}
            <button
              class="w-full flex items-center gap-2 px-3 py-2 text-left text-sm hover:bg-ctp-surface0/30 transition-colors
                {selectedWorkspace?.id === w.id
                ? 'text-ctp-blue'
                : 'text-ctp-text'}"
              onclick={() => selectWorkspace(w)}
            >
              <Folder size={14} class="shrink-0 text-ctp-overlay0" />
              <span class="truncate">{w.name}</span>
            </button>
          {/each}
          {#if filteredWorkspaces.length === 0}
            <div class="px-3 py-4 text-center text-ctp-subtext0 text-xs">
              no workspaces found
            </div>
          {/if}
        </div>
        <div class="p-2 border-t border-ctp-surface0/30">
          <input
            bind:value={workspaceQuery}
            type="text"
            placeholder="Search workspaces..."
            class="w-full bg-ctp-surface0/20 border border-ctp-surface0/30 px-2 py-1.5 text-xs text-ctp-text placeholder-ctp-subtext0 focus:outline-none focus:ring-1 focus:ring-ctp-blue/30"
          />
        </div>
      </div>
    {/if}
  </div>

  {#if selectedWorkspace}
    <ChevronRight size={14} class="text-ctp-overlay0 shrink-0" />

    <!-- Experiment segment -->
    <div class="relative">
      <button
        class="flex items-center gap-1.5 px-2 py-1 hover:bg-ctp-surface0/30 transition-colors text-ctp-text"
        onclick={toggleExperimentDropdown}
      >
        <FlaskConical size={14} class="text-ctp-overlay0" />
        <span class="max-w-[200px] truncate"
          >{selectedExperiment?.name ?? "Experiments"}</span
        >
        <ChevronUp size={12} class="text-ctp-overlay0" />
      </button>

      {#if showExperimentDropdown}
        <div
          class="absolute bottom-full left-0 mb-1 w-72 bg-ctp-mantle border border-ctp-surface0/50 shadow-xl shadow-ctp-crust/40 z-50"
        >
          <div class="max-h-[300px] overflow-y-auto py-1">
            {#each filteredExperiments as exp}
              <button
                class="w-full flex items-center gap-2 px-3 py-2 text-left text-sm hover:bg-ctp-surface0/30 transition-colors
                  {selectedExperiment?.id === exp.id
                  ? 'text-ctp-blue'
                  : 'text-ctp-text'}"
                onclick={() => selectExperiment(exp)}
              >
                <FlaskConical size={14} class="shrink-0 text-ctp-overlay0" />
                <div class="flex-1 min-w-0">
                  <div class="truncate">{exp.name}</div>
                  {#if exp.tags?.length}
                    <div class="text-[10px] text-ctp-overlay0 truncate">
                      {exp.tags.map((t) => `#${t}`).join(" ")}
                    </div>
                  {/if}
                </div>
              </button>
            {/each}
            {#if filteredExperiments.length === 0}
              <div class="px-3 py-4 text-center text-ctp-subtext0 text-xs">
                no experiments found
              </div>
            {/if}
          </div>
          <div class="p-2 border-t border-ctp-surface0/30">
            <input
              bind:value={experimentQuery}
              type="text"
              placeholder="Search experiments..."
              class="w-full bg-ctp-surface0/20 border border-ctp-surface0/30 px-2 py-1.5 text-xs text-ctp-text placeholder-ctp-subtext0 focus:outline-none focus:ring-1 focus:ring-ctp-blue/30"
            />
          </div>
        </div>
      {/if}
    </div>
  {/if}
</div>
