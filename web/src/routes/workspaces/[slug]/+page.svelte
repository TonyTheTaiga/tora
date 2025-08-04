<script lang="ts">
  import {
    getCreateExperimentModal,
    getExperimentToEdit,
    getExperimentToDelete,
    openCreateExperimentModal,
  } from "$lib/state/app.svelte.js";
  import {
    CreateExperimentModal,
    DeleteConfirmationModal,
    EditExperimentModal,
  } from "$lib/components/modals";
  import { PageHeader, SearchInput } from "$lib/components";
  import { ExperimentList } from "$lib/components/lists";
  import { Plus } from "@lucide/svelte";

  import type { Experiment } from "$lib/types";

  let { data } = $props();

  let currentWorkspace = $derived(data.currentWorkspace);
  let experiments: Experiment[] = $derived(data.experiments || []);
  let searchQuery = $state("");
  let createExperimentModal = $derived(getCreateExperimentModal());
  let editExperimentModal = $derived(getExperimentToEdit());
  let deleteExperimentModal = $derived(getExperimentToDelete());

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
  }
</script>

{#if createExperimentModal}
  <CreateExperimentModal workspace={currentWorkspace} />
{/if}

{#if deleteExperimentModal}
  <DeleteConfirmationModal
    experiment={deleteExperimentModal}
    bind:experiments
  />
{/if}

{#if editExperimentModal}
  <EditExperimentModal bind:experiment={editExperimentModal} />
{/if}

<div class="font-mono">
  <!-- Header -->
  <PageHeader
    title={currentWorkspace?.name || "Loading..."}
    description={currentWorkspace?.description || undefined}
    subtitle="{experiments.length} experiment{experiments.length !== 1
      ? 's'
      : ''}"
    additionalInfo={currentWorkspace?.id || undefined}
    onAdditionalInfoClick={currentWorkspace?.id
      ? () => copyToClipboard(currentWorkspace.id)
      : undefined}
    additionalInfoTitle="click to copy workspace id"
  >
    {#snippet actionButton()}
      <button
        onclick={() => openCreateExperimentModal()}
        class="group relative bg-ctp-surface0/20 backdrop-blur-md border border-ctp-surface0/30 text-ctp-text hover:bg-ctp-surface0/30 hover:border-ctp-surface0/50 px-3 py-2 md:px-4 text-sm font-mono transition-all flex-shrink-0"
      >
        <div class="flex items-center gap-2">
          <Plus class="w-4 h-4" />
          <span class="hidden sm:inline">New</span>
        </div>
      </button>
    {/snippet}
  </PageHeader>

  <!-- Search and filter bar -->
  <SearchInput bind:value={searchQuery} placeholder="search experiments..." />

  <!-- Terminal-style experiments display -->
  <div class="font-mono">
    {#if experiments.length === 0}
      <div class="space-y-3 text-base">
        <div class="text-ctp-subtext0 text-sm">
          no experiments found in this workspace
        </div>
        <div class="mt-4">
          <button
            onclick={() => openCreateExperimentModal()}
            class="text-ctp-mauve hover:text-ctp-mauve/80 transition-colors text-sm"
          >
            [create experiment]
          </button>
        </div>
      </div>
    {:else}
      <ExperimentList {experiments} {searchQuery} {formatDate} />
    {/if}
  </div>
</div>
