<script lang="ts">
  import {
    getCreateExperimentModal,
    getEditExperimentModal,
    getDeleteExperimentModal,
  } from "$lib/state/app.svelte.js";
  import ExperimentsList from "./experiments-list.svelte";
  import CreateExperimentModal from "./create-experiment-modal.svelte";
  import DeleteConfirmationModal from "./delete-confirmation-modal.svelte";
  import EditExperimentModal from "./edit-experiment-modal.svelte";

  let { data = $bindable() } = $props();
  let { workspace } = $derived(data);
  let experiments = $state(data.experiments);

  $effect(() => {
    experiments = data.experiments;
  });

  let createExperimentModal = $derived(getCreateExperimentModal());
  let editExperimentModal = $derived(getEditExperimentModal());
  let deleteExperimentModal = $derived(getDeleteExperimentModal());
</script>

{#if createExperimentModal}
  <CreateExperimentModal {workspace} {experiments} />
{/if}

{#if deleteExperimentModal}
  <DeleteConfirmationModal
    experiment={deleteExperimentModal}
    bind:experiments
  />
{/if}

{#if editExperimentModal}
  <EditExperimentModal experiment={editExperimentModal} {workspace} {experiments} />
{/if}

<div class="p-4">
  <ExperimentsList bind:experiments />
</div>
