<script lang="ts">
  import {
    getCreateExperimentModal,
    getEditExperimentModal,
    getDeleteExperimentModal,
    openCreateExperimentModal,
  } from "$lib/state/app.svelte.js";
  import CreateExperimentModal from "$lib/components/modals/create-experiment-modal.svelte";
  import DeleteConfirmationModal from "$lib/components/modals/delete-confirmation-modal.svelte";
  import EditExperimentModal from "$lib/components/modals/edit-experiment-modal.svelte";
  import ExperimentsListMobile from "./experiments-list-mobile.svelte";
  import ExperimentsListDesktop from "./experiments-list-desktop.svelte";
  import { Plus } from "@lucide/svelte";
  import { onMount } from "svelte";

  // Mock data for now - will be replaced with Rust backend data
  let currentWorkspace = $state({
    id: "1",
    name: "ML Research",
    description: "Machine learning experiments and research",
    role: "OWNER"
  });

  let experiments = $state([
    {
      id: "exp_1",
      name: "Baseline Model",
      description: "Initial baseline experiment",
      hyperparams: [],
      tags: ["baseline", "initial"],
      createdAt: new Date(),
      updatedAt: new Date(),
      availableMetrics: ["accuracy", "loss"],
      workspaceId: "1"
    }
  ]);
  let searchQuery = $state("");
  let copiedId = $state(false);
  let createExperimentModal = $derived(getCreateExperimentModal());
  let editExperimentModal = $derived(getEditExperimentModal());
  let deleteExperimentModal = $derived(getDeleteExperimentModal());

  let filteredExperiments = $derived(
    experiments
      .map((exp) => ({
        exp,
        name: exp.name.toLowerCase(),
        desc: exp.description?.toLowerCase() ?? "",
        tags: exp.tags?.map((t) => t.toLowerCase()) ?? [],
      }))
      .filter((entry) => {
        const q = searchQuery.toLowerCase();
        return entry.name.includes(q);
      })
      .map((e) => e.exp),
  );

  function formatDate(date: Date): string {
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year:
        date.getFullYear() !== new Date().getFullYear() ? "numeric" : undefined,
    });
  }

  function copyToClipboard(id: string) {
    navigator.clipboard.writeText(id);
    copiedId = true;
    setTimeout(() => (copiedId = false), 1200);
  }

  const handleKeydown = (event: KeyboardEvent) => {
    if (event.key === "/") {
      event.preventDefault();
      const searchElement = document.querySelector<HTMLInputElement>(
        'input[type="search"]',
      );
      searchElement?.focus();
    }
  };

  onMount(() => {
    window.addEventListener("keydown", handleKeydown);

    return () => {
      window.removeEventListener("keydown", handleKeydown);
    };
  });

  // No longer needed - using static mock data
</script>

{#if createExperimentModal}
  <CreateExperimentModal workspace={currentWorkspace} />
{/if}

{#if deleteExperimentModal}
  <DeleteConfirmationModal
    experiment={deleteExperimentModal}
    bind:experiments
    onDelete={async (experimentId) => {
      // This would need to be implemented with a server action
      // For now, just throw an error to indicate it needs implementation
      throw new Error("Delete experiment functionality needs to be implemented with server actions");
    }}
  />
{/if}

{#if editExperimentModal}
  <EditExperimentModal bind:experiment={editExperimentModal} />
{/if}

<div class="font-mono">
  <!-- Header -->
  <div
    class="flex items-center justify-between p-4 md:p-6 border-b border-ctp-surface0/10"
  >
    <div
      class="flex items-stretch gap-3 md:gap-4 min-w-0 flex-1 pr-4 min-h-fit"
    >
      <div
        class="w-2 bg-ctp-blue rounded-full flex-shrink-0 self-stretch"
      ></div>
      <div class="min-w-0 flex-1 py-1">
        <h1 class="text-lg md:text-xl text-ctp-text truncate font-mono">
          {currentWorkspace?.name || "Workspace"}
        </h1>
        <div class="text-sm text-ctp-subtext0 space-y-1">
          <div>
            {#if currentWorkspace?.description}
              <span>{currentWorkspace.description}</span>
            {/if}
          </div>
          <div>
            {experiments.length} experiment{experiments.length !== 1 ? "s" : ""}
          </div>
          {#if currentWorkspace?.id}
            <button
              onclick={() => copyToClipboard(currentWorkspace.id)}
              class="text-ctp-blue hover:text-ctp-blue/80 transition-colors flex text-start"
              title="click to copy workspace id"
            >
              <span>{currentWorkspace.id}</span>
            </button>
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
      <div
        class="flex items-center bg-ctp-surface0/20 focus-within:ring-1 focus-within:ring-ctp-text/20 transition-all"
      >
        <span class="text-ctp-subtext0 font-mono text-base px-4 py-3">/</span>
        <input
          type="search"
          placeholder="search experiments..."
          bind:value={searchQuery}
          class="flex-1 bg-transparent border-0 py-3 pr-4 text-ctp-text placeholder-ctp-subtext0 focus:outline-none font-mono text-base"
        />
      </div>
    </div>
  </div>

  <!-- Terminal-style experiments display -->
  <div class="px-4 md:px-6 font-mono">
    {#if filteredExperiments.length === 0 && searchQuery}
      <div class="text-ctp-subtext0 text-base">
        <div>search "{searchQuery}"</div>
        <div class="text-ctp-subtext1 ml-2">no results found</div>
      </div>
    {:else if experiments.length === 0}
      <div class="space-y-3 text-base">
        <div class="text-ctp-subtext0 text-sm">
          no experiments found in this workspace
        </div>
        <div class="mt-4">
          <button
            onclick={() => openCreateExperimentModal()}
            class="text-ctp-blue hover:text-ctp-blue/80 transition-colors text-sm"
          >
            [create experiment]
          </button>
        </div>
      </div>
    {:else}
      <!-- Responsive experiment layouts -->
      <ExperimentsListMobile experiments={filteredExperiments} {formatDate} />

      <ExperimentsListDesktop experiments={filteredExperiments} {formatDate} />

      <!-- Summary line -->
      <div
        class="flex items-center text-sm text-ctp-subtext0 pt-2 border-t border-ctp-surface0/20 mt-4"
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
