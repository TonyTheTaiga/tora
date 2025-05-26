<script lang="ts">
  import CreateExperimentModal from "$lib/components/create-experiment-modal.svelte";
  import ExperimentsList from "$lib/components/experiments-list.svelte";
  import type { Experiment } from "$lib/types";
  import type { PageData } from "./$types";
  import Toolbar from "$lib/components/toolbar.svelte";
  import DeleteConfirmationModal from "$lib/components/delete-confirmation-modal.svelte";
  import EditExperimentModal from "$lib/components/edit-experiment-modal.svelte";

  let { data }: { data: PageData } = $props();

  let experiments: Experiment[] = $derived(data.experiments);
  let hasExperiments: boolean = $derived(experiments.length > 0);

  let modalState = $state({
    createExperiment: false,
    selectedForDelete: null as Experiment | null,
    selectedForEdit: null as Experiment | null,
    selectedExperiment: null as Experiment | null,
  });
</script>

{#if modalState.createExperiment}
  <CreateExperimentModal
    bind:createNewExperimentFlag={modalState.createExperiment}
  />
{/if}

{#if modalState.selectedForDelete}
  <DeleteConfirmationModal
    bind:experiment={modalState.selectedForDelete}
    bind:experiments
  />
{/if}

{#if modalState.selectedForEdit}
  <EditExperimentModal bind:experiment={modalState.selectedForEdit} />
{/if}

<Toolbar
  bind:selectedExperiment={modalState.selectedExperiment}
  bind:isOpenCreate={modalState.createExperiment}
  {hasExperiments}
/>

<ExperimentsList
  bind:experiments
  bind:createNewExperimentFlag={modalState.createExperiment}
  bind:selectedForEdit={modalState.selectedForEdit}
  bind:selectedForDelete={modalState.selectedForDelete}
  bind:selectedExperiment={modalState.selectedExperiment}
></ExperimentsList>
