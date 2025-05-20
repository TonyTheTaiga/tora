<script lang="ts">
  import type { Experiment } from "$lib/types";
  import NewExperimentCard from "./new-experiment-card.svelte";
  import ExperimentSimple from "./experiment-simple.svelte";
  import ExperimentDetailed from "./experiment-detailed.svelte";
  import DeleteConfirmationModal from "./delete-confirmation-modal.svelte";
  import { page } from "$app/state";

  let isUserSignedIn: boolean = $state(!!page.data.session);
  let {
    experiments = $bindable(),
    isOpen = $bindable(),
  }: { experiments: Experiment[]; isOpen: boolean } = $props();
  let selectedId = $state<string | null>(null);
  let highlighted = $state<string[]>([]);
  let selectedForDelete = $state<Experiment | null>(null);
</script>

<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
  <NewExperimentCard bind:isUserSignedIn bind:isOpen />

  {#each experiments as experiment, idx (experiment.id)}
    {#if selectedId !== experiment.id}
      <ExperimentSimple
        bind:selectedId
        bind:highlighted
        bind:selectedForDelete
        {experiment}
      />
    {:else}
      <ExperimentDetailed
        bind:selectedId
        bind:highlighted
        bind:experiment={experiments[idx]}
        bind:selectedForDelete
      />
    {/if}
  {/each}
</div>

<DeleteConfirmationModal bind:experiment={selectedForDelete} bind:experiments />
