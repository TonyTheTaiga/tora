<script lang="ts">
  import CreateExperimentModal from "$lib/components/create-experiment-modal.svelte";
  import ExperimentsList from "$lib/components/experiments-list.svelte";
  import type { Experiment } from "$lib/types";
  import type { PageData } from "./$types";
  import Toolbar from "$lib/components/toolbar.svelte";
  import DeleteConfirmationModal from "$lib/components/delete-confirmation-modal.svelte";
  import EditExperimentModal from "$lib/components/edit-experiment-modal.svelte";

  let { data }: { data: PageData } = $props();
  let experiments: Experiment[] = $state(data.experiments);
  let createNewExperimentFlag: boolean = $state(false);
  let selectedForDelete: Experiment | null = $state(null);
  let selectedForEdit: Experiment | null = $state(null);
  let selectedExperiment: Experiment | null = $state(null);
</script>

{#if createNewExperimentFlag}
  <CreateExperimentModal bind:createNewExperimentFlag />
{/if}

{#if selectedForDelete}
  <DeleteConfirmationModal
    bind:experiment={selectedForDelete}
    bind:experiments
  />
{/if}

{#if selectedForEdit}
  <EditExperimentModal bind:experiment={selectedForEdit} />
{/if}

<Toolbar bind:selectedExperiment />

<ExperimentsList
  bind:experiments
  bind:createNewExperimentFlag
  bind:selectedForEdit
  bind:selectedForDelete
  bind:selectedExperiment
></ExperimentsList>
