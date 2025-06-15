<script lang="ts">
  import {
    getCreateExperimentModal,
    getEditExperimentModal,
    getDeleteExperimentModal,
    openCreateExperimentModal,
  } from "$lib/state/app.svelte.js";
  import CreateExperimentModal from "./create-experiment-modal.svelte";
  import DeleteConfirmationModal from "./delete-confirmation-modal.svelte";
  import EditExperimentModal from "./edit-experiment-modal.svelte";
  import ExperimentsListMobile from "./experiments-list-mobile.svelte";
  import ExperimentsListDesktop from "./experiments-list-desktop.svelte";
  import type { Experiment } from "$lib/types";
  import { Plus, Copy, ClipboardCheck } from "lucide-svelte";

  let { data = $bindable() } = $props();
  let { currentWorkspace } = $derived(data);
  let experiments = $state(data.experiments);
  let searchQuery = $state("");
  let debouncedQuery = $state("");
  let highlighted = $state<string[]>([]);
  let copiedId = $state(false);

  $effect(() => {
    experiments = data.experiments;
  });

  let normalized = $derived(
    experiments.map((exp) => ({
      exp,
      name: exp.name.toLowerCase(),
      desc: exp.description?.toLowerCase() ?? "",
      tags: exp.tags?.map((t) => t.toLowerCase()) ?? [],
    })),
  );

  let filteredExperiments = $derived(
    normalized
      .filter((entry) => {
        const q = debouncedQuery.toLowerCase();
        return (
          entry.name.includes(q) ||
          entry.desc.includes(q) ||
          entry.tags.some((t) => t.includes(q))
        );
      })
      .map((e) => e.exp),
  );

  $effect(() => {
    const id = setTimeout(() => (debouncedQuery = searchQuery), 200);
    return () => clearTimeout(id);
  });

  let createExperimentModal = $derived(getCreateExperimentModal());
  let editExperimentModal = $derived(getEditExperimentModal());
  let deleteExperimentModal = $derived(getDeleteExperimentModal());

  function formatDate(date: Date): string {
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year:
        date.getFullYear() !== new Date().getFullYear() ? "numeric" : undefined,
    });
  }

  async function toggleHighlight(experiment: Experiment) {
    if (highlighted.includes(experiment.id)) {
      highlighted = [];
    } else {
      try {
        const response = await fetch(`/api/experiments/${experiment.id}/ref`);
        if (!response.ok) return;
        const data = await response.json();
        highlighted = [...data, experiment.id];
      } catch (err) {}
    }
  }

  function copyToClipboard(id: string) {
    navigator.clipboard.writeText(id);
    copiedId = true;
    setTimeout(() => (copiedId = false), 1200);
  }
</script>

{#if createExperimentModal}
  <CreateExperimentModal workspace={currentWorkspace} {experiments} />
{/if}

{#if deleteExperimentModal}
  <DeleteConfirmationModal
    experiment={deleteExperimentModal}
    bind:experiments
  />
{/if}

{#if editExperimentModal}
  <EditExperimentModal bind:experiment={editExperimentModal} bind:experiments />
{/if}

<div class="bg-ctp-base font-mono">
  <!-- Header -->
  <div
    class="flex items-center justify-between p-4 md:p-6 border-b border-ctp-surface0/10"
  >
    <div class="flex items-center gap-3 md:gap-4 min-w-0 flex-1 pr-4">
      <div class="w-2 h-6 md:h-8 bg-ctp-blue rounded-full flex-shrink-0"></div>
      <div class="min-w-0 flex-1">
        <h1 class="text-lg md:text-xl text-ctp-text truncate font-mono">
          {currentWorkspace?.name || "Workspace"}
        </h1>
        <div class="text-xs text-ctp-subtext0 space-y-1">
          <div>
            {experiments.length} experiment{experiments.length !== 1 ? "s" : ""}
            {#if currentWorkspace?.description}
              <span class="hidden sm:inline"
                >• {currentWorkspace.description}</span
              >
            {/if}
          </div>
          {#if currentWorkspace?.id}
            <div class="flex items-center gap-2">
              <span>id:</span>
              <button
                onclick={() => copyToClipboard(currentWorkspace.id)}
                class="text-ctp-blue hover:text-ctp-blue/80 transition-colors flex items-center gap-1"
                title="click to copy workspace id"
              >
                <span>{currentWorkspace.id}</span>
                {#if copiedId}
                  <ClipboardCheck size={10} class="text-ctp-green" />
                {:else}
                  <Copy size={10} />
                {/if}
              </button>
            </div>
          {/if}
        </div>
      </div>
    </div>

    <button
      onclick={() => openCreateExperimentModal()}
      class="group relative bg-ctp-surface0/20 backdrop-blur-md border border-ctp-surface0/30 text-ctp-text hover:bg-ctp-surface0/30 hover:border-ctp-surface0/50 px-3 py-2 md:px-4 text-sm font-mono transition-all flex-shrink-0"
    >
      <div class="flex items-center gap-2">
        <Plus class="w-4 h-4" />
        <span class="hidden sm:inline">new</span>
      </div>
    </button>
  </div>

  <!-- Search and filter bar -->
  <div class="px-4 md:px-6 py-4">
    <div class="max-w-lg">
      <div class="relative">
        <input
          type="text"
          placeholder="Search experiments..."
          bind:value={searchQuery}
          class="w-full bg-ctp-surface0/20 border-0 px-4 py-3 text-ctp-text placeholder-ctp-subtext0 focus:outline-none focus:ring-1 focus:ring-ctp-text/20 transition-all font-mono text-sm"
        />
        <div
          class="absolute right-3 top-1/2 transform -translate-y-1/2 text-xs text-ctp-subtext0 font-mono"
        >
          {filteredExperiments.length}/{experiments.length}
        </div>
      </div>
    </div>
  </div>

  <!-- Terminal-style experiments display -->
  <div class="px-4 md:px-6 font-mono">
    {#if filteredExperiments.length === 0 && searchQuery}
      <div class="text-ctp-subtext0 text-sm">
        <div>$ search "{searchQuery}"</div>
        <div class="text-ctp-subtext1 ml-2">no results found</div>
      </div>
    {:else if experiments.length === 0}
      <div class="space-y-3 text-sm">
        <div class="text-ctp-subtext0 text-xs">
          no experiments found in this workspace
        </div>
        <div class="mt-4">
          <button
            onclick={() => openCreateExperimentModal()}
            class="text-ctp-blue hover:text-ctp-blue/80 transition-colors text-xs"
          >
            [create experiment]
          </button>
        </div>
      </div>
    {:else}
      <!-- Responsive experiment layouts -->
      <ExperimentsListMobile
        experiments={filteredExperiments}
        {highlighted}
        onToggleHighlight={toggleHighlight}
        {formatDate}
      />

      <ExperimentsListDesktop
        experiments={filteredExperiments}
        {highlighted}
        onToggleHighlight={toggleHighlight}
        {formatDate}
      />

      <!-- Summary line -->
      <div
        class="flex items-center text-xs text-ctp-subtext0 pt-2 border-t border-ctp-surface0/20 mt-4"
      >
        <div class="flex-1">
          {filteredExperiments.length} experiment{filteredExperiments.length !==
          1
            ? "s"
            : ""} total
        </div>
      </div>
    {/if}
  </div>
</div>
