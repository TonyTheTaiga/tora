<script lang="ts">
  import {
    getCreateExperimentModal,
    getEditExperimentModal,
    getDeleteExperimentModal,
    openCreateExperimentModal,
  } from "$lib/state/app.svelte.js";
  import {
    CreateExperimentModal,
    DeleteConfirmationModal,
    EditExperimentModal,
  } from "$lib/components/modals";
  import { PageHeader } from "$lib/components";
  import { ExperimentList } from "$lib/components/lists";
  import { Plus } from "@lucide/svelte";
  import { onMount } from "svelte";

  import type { Experiment } from "$lib/types";

  let { data } = $props();

  let currentWorkspace = $derived(data.currentWorkspace);
  let experiments: Experiment[] = $derived(data.experiments || []);
  let searchQuery = $state("");
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

  function copyToClipboard(id: string) {
    navigator.clipboard.writeText(id);
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
          <span class="hidden sm:inline">new</span>
        </div>
      </button>
    {/snippet}
  </PageHeader>

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
    {#if experiments.length === 0}
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
      <ExperimentList {experiments} {searchQuery} {formatDate} />
    {/if}
  </div>
</div>
