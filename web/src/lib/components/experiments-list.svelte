<script lang="ts">
  import type { Experiment } from "$lib/types";
  import NewExperimentCard from "./new-experiment-card.svelte";
  import ExperimentSimple from "./experiment-simple.svelte";
  import ExperimentDetailed from "./experiment-detailed.svelte";
  import { page } from "$app/state";

  let {
    experiments = $bindable(),
    createNewExperimentFlag = $bindable(),
    selectedForEdit = $bindable(),
    selectedForDelete = $bindable(),
    selectedExperiment = $bindable(),
  }: {
    experiments: Experiment[];
    createNewExperimentFlag: boolean;
    selectedForDelete: Experiment | null;
    selectedForEdit: Experiment | null;
    selectedExperiment: Experiment | null;
  } = $props();

  let isUserSignedIn: boolean = $state(!!page.data.session);
  let highlighted = $state<string[]>([]);
</script>

<div class="grid grid-cols-1 gap-4">
  <NewExperimentCard bind:isUserSignedIn bind:createNewExperimentFlag />

  {#each experiments as experiment, idx (experiment.id)}
    <div id={`experiment-${experiment.id}`}>
      {#if !selectedExperiment || selectedExperiment.id !== experiment.id}
        <div class="hover:border rounded-lg">
          <ExperimentSimple
            bind:selectedExperiment
            bind:highlighted
            bind:selectedForDelete
            {experiment}
          />
        </div>
      {:else}
        <div class="h-full">
          <ExperimentDetailed
            bind:selectedExperiment
            bind:highlighted
            bind:experiment={experiments[idx]}
            bind:selectedForDelete
            bind:selectedForEdit
          />
        </div>
      {/if}
    </div>
  {/each}
</div>
