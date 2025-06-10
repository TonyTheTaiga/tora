<script lang="ts">
  import type { PageData } from "./$types";
  import CreateExperimentModal from "$lib/components/create-experiment-modal.svelte";
  import ExperimentsList from "$lib/components/experiments-list.svelte";
  import Toolbar from "$lib/components/toolbar.svelte";
  import DeleteConfirmationModal from "$lib/components/delete-confirmation-modal.svelte";
  import EditExperimentModal from "$lib/components/edit-experiment-modal.svelte";
  import LandingPage from "$lib/components/landing-page.svelte";
  import ComparisonToolbar from "$lib/components/comparison/comparison-toolbar.svelte";
  import { page } from "$app/state";
  import { getMode } from "$lib/state/comparison.svelte.js";
  import {
    getCreateExperimentModal,
    getEditExperimentModal,
    getDeleteExperimentModal,
  } from "$lib/state/app.svelte.js";

  let user = $derived(page.data.user);
  let { data = $bindable() }: { data: PageData } = $props();
  let experiments = $derived([...data.experiments]);

  let hasExperiments: boolean = $derived(experiments.length > 0);
  let createExperimentModal = $derived(getCreateExperimentModal());
  let editExperimentModal = $derived(getEditExperimentModal());
  let deleteExperimentModal = $derived(getDeleteExperimentModal());
</script>

{#if user}
  {#if createExperimentModal}
    <CreateExperimentModal />
  {/if}

  {#if deleteExperimentModal}
    <DeleteConfirmationModal
      experiment={deleteExperimentModal}
      bind:experiments
    />
  {/if}

  {#if editExperimentModal}
    <EditExperimentModal experiment={editExperimentModal} />
  {/if}

  <Toolbar {hasExperiments} />

  <div class="pt-4 px-0 sm:px-2 md:px-4">
    {#if getMode()}
      <div class="sticky top-0 z-30 flex justify-center mb-4 bg-ctp-base py-2">
        <ComparisonToolbar />
      </div>
    {/if}

    <ExperimentsList bind:experiments />
  </div>
{:else}
  <LandingPage />
{/if}
