<script lang="ts">
  import type { Experiment } from "$lib/types";
  import NewExperimentCard from "./new-experiment-card.svelte";
  import ExperimentSimple from "./experiment-simple.svelte";
  import ExperimentDetailed from "./experiment-detailed.svelte";
  import { page } from "$app/state";

  let isUserSignedIn: boolean = $state(!!page.data.session);
  let {
    experiments = $bindable(),
    isOpen = $bindable(),
  }: { experiments: Experiment[]; isOpen: boolean } = $props();
  let selectedId = $state<string | null>(null);
  let highlighted = $state<string[]>([]);
</script>

<div
  class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 grid-rows-[200px]"
>
  <NewExperimentCard bind:isUserSignedIn bind:isOpen />

  {#each experiments as experiment, idx (experiment.id)}
    {#if selectedId !== experiment.id}
      <ExperimentSimple bind:selectedId bind:highlighted {experiment} />
    {:else}
      <ExperimentDetailed
        bind:selectedId
        bind:highlighted
        bind:experiment={experiments[idx]}
      />
    {/if}
  {/each}
</div>
