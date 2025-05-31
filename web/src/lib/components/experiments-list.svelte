<script lang="ts">
  import type { Experiment } from "$lib/types";
  import ExperimentSimple from "./experiment-simple.svelte";
  import ExperimentDetailed from "./experiment-detailed.svelte";
  import type { Attachment } from "svelte/attachments";
  import {
    toggleMode,
    getMode,
  } from "$lib/components/comparison/state.svelte.js";

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

  const focusOnExpandAttatchment: Attachment = (element) => {
    element.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });

    return () => {};
  };
</script>

<div
  class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 [&>*:has(.expanded-experiment)]:md:col-span-2 [&>*:has(.expanded-experiment)]:lg:col-span-3"
>
  {#each experiments as experiment, idx (experiment.id)}
    <div id={`experiment-${experiment.id}`}>
      {#if !selectedExperiment || selectedExperiment.id !== experiment.id}
        <div
          class="rounded-lg bg-ctp-mantle overflow-hidden h-[200px] p-4 hover:bg-ctp-surface0 transition-colors cursor-pointer
            {highlighted.length > 0 && !highlighted.includes(experiment.id)
            ? 'opacity-40'
            : ''}"
          role="button"
          tabindex="0"
          onclick={() => {
            if (getMode()) {
            } else {
              selectedExperiment = experiment;
            }
          }}
          onkeydown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              selectedExperiment = experiment;
            }
          }}
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
          {@attach focusOnExpandAttatchment}
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
