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
  let recentlyMinimized = $state<string | null>(null);
</script>

<div
  class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 [&>*:has(.expanded-experiment)]:md:col-span-2 [&>*:has(.expanded-experiment)]:lg:col-span-3"
>
  <NewExperimentCard bind:isUserSignedIn bind:createNewExperimentFlag />

  {#each experiments as experiment, idx (experiment.id)}
    <div
      id={`experiment-${experiment.id}`}
      class={`transition-all duration-400 ease-in-out transform ${selectedExperiment && selectedExperiment.id === experiment.id ? "h-full" : ""}`}
    >
      {#if !selectedExperiment || selectedExperiment.id !== experiment.id}
        <div
          class="hover:border rounded-lg
          {recentlyMinimized === experiment.id ? 'shadow-highlight' : ''}"
          onanimationend={() => {
            if (recentlyMinimized === experiment.id) {
              recentlyMinimized = null;
            }
          }}
        >
          <ExperimentSimple
            bind:selectedExperiment
            bind:highlighted
            bind:selectedForDelete
            bind:recentlyMinimized
            {experiment}
          />
        </div>
      {:else}
        <div
          class="expanded-experiment h-full animate-expand will-change-opacity-transform"
        >
          <ExperimentDetailed
            bind:selectedExperiment
            bind:highlighted
            bind:experiment={experiments[idx]}
            bind:selectedForDelete
            bind:recentlyMinimized
            bind:selectedForEdit
          />
        </div>
      {/if}
    </div>
  {/each}
</div>

<style>
  @keyframes shadow-glow {
    0% {
      box-shadow: 0 0 0px 0px rgba(183, 189, 248, 0);
      transform: scale(1);
    }
    15% {
      box-shadow: 0 0 12px 4px rgba(183, 189, 248, 0.5);
      transform: scale(1.01);
    }
    60% {
      box-shadow: 0 0 8px 2px rgba(183, 189, 248, 0.3);
      transform: scale(1.005);
    }
    100% {
      box-shadow: 0 0 0px 0px rgba(183, 189, 248, 0);
      transform: scale(1);
    }
  }

  .will-change-opacity-transform {
    will-change: opacity, transform;
  }

  .shadow-highlight {
    animation: shadow-glow 1.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    overflow: hidden;
    will-change: transform, box-shadow;
  }

  @keyframes expand {
    from {
      opacity: 0.4;
      transform: scale(0.95);
    }
    to {
      opacity: 1;
      transform: scale(1);
    }
  }

  .animate-expand {
    animation: expand 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    will-change: transform, opacity;
  }
</style>
