<script lang="ts">
  import type { Experiment } from "$lib/types";
  import NewExperimentCard from "./new-experiment-card.svelte";
  import ExperimentSimple from "./experiment-simple.svelte";
  import ExperimentDetailed from "./experiment-detailed.svelte";
  import DeleteConfirmationModal from "./delete-confirmation-modal.svelte";
  import EditExperimentModal from "./edit-experiment-modal.svelte";

  import { page } from "$app/state";

  let {
    experiments = $bindable(),
    isOpen = $bindable(),
  }: { experiments: Experiment[]; isOpen: boolean } = $props();
  let isUserSignedIn: boolean = $state(!!page.data.session);
  let selectedId = $state<string | null>(null);
  let highlighted = $state<string[]>([]);
  let selectedForDelete = $state<Experiment | null>(null);
  let selectedForEdit = $state<Experiment | null>(null);
  let recentlyMinimized = $state<string | null>(null);
</script>

{#if selectedForDelete}
  <DeleteConfirmationModal
    bind:experiment={selectedForDelete}
    bind:experiments
  />
{/if}

{#if selectedForEdit}
  <EditExperimentModal bind:experiment={selectedForEdit} />
{/if}

<div
  class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 [&>*:has(.expanded-experiment)]:md:col-span-2 [&>*:has(.expanded-experiment)]:lg:col-span-3"
>
  <NewExperimentCard bind:isUserSignedIn bind:isOpen />

  {#each experiments as experiment, idx (experiment.id)}
    <div
      id={`experiment-${experiment.id}`}
      class={`transition-all duration-300 ${selectedId === experiment.id ? "h-full" : ""}`}
    >
      {#if selectedId !== experiment.id}
        <div
          class={recentlyMinimized === experiment.id ? "shadow-highlight" : ""}
          onanimationend={() => {
            if (recentlyMinimized === experiment.id) {
              recentlyMinimized = null;
            }
          }}
        >
          <ExperimentSimple
            bind:selectedId
            bind:highlighted
            bind:selectedForDelete
            bind:recentlyMinimized
            {experiment}
          />
        </div>
      {:else}
        <div class="expanded-experiment h-full animate-expand">
          <ExperimentDetailed
            bind:selectedId
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
    }
    15% {
      box-shadow: 0 0 10px 3px rgba(183, 189, 248, 0.4);
    }
    60% {
      box-shadow: 0 0 8px 2px rgba(183, 189, 248, 0.3);
    }
    100% {
      box-shadow: 0 0 0px 0px rgba(183, 189, 248, 0);
    }
  }

  .shadow-highlight {
    animation: shadow-glow 2s ease-out forwards;
    border-radius: 0.5rem; /* Matches rounded-lg */
    overflow: hidden;
  }
</style>
