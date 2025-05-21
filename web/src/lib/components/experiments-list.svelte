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

  let highlighted = $state<string[]>([]);
</script>

<div
  class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 [&>*:has(.expanded-experiment)]:md:col-span-2 [&>*:has(.expanded-experiment)]:lg:col-span-3"
>
  <div
    class="h-[200px] rounded-lg bg-ctp-crust border-2 border-dashed border-ctp-subtext0 p-4"
  >
    <NewExperimentCard bind:createNewExperimentFlag />
  </div>

  {#each experiments as experiment, idx (experiment.id)}
    <div id={`experiment-${experiment.id}`}>
      {#if !selectedExperiment || selectedExperiment.id !== experiment.id}
        <div
          class="rounded-lg bg-ctp-mantle overflow-hidden h-[200px] p-4
            {highlighted.length > 0 && !highlighted.includes(experiment.id)
            ? 'opacity-40'
            : ''}"
        >
          <ExperimentSimple
            bind:selectedExperiment
            bind:highlighted
            bind:selectedForDelete
            {experiment}
          />
        </div>
      {:else}
        <div
          class="rounded-lg overflow-hidden bg-ctp-base expanded-experiment
            {highlighted.length > 0 && !highlighted.includes(experiment.id)
            ? 'opacity-40'
            : ''}"
        >
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
