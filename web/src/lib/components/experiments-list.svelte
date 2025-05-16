<script lang="ts">
  import type { Experiment } from "$lib/types";
  import ExperimentSimple from "./experiment-simple.svelte";
  import ExperimentDetailed from "./experiment-detailed.svelte";

  const {
    experiments = $bindable(),
    children,
  }: { experiments: Experiment[]; children } = $props();
  let selectedId = $state<string | null>(null);
  let highlighted = $state<string[]>([]);
</script>

<section>
  <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
    {@render children()}

    {#each experiments as experiment, idx (experiment.id)}
      <div
        class="
            rounded-lg border border-ctp-surface1 overflow-hidden
            {selectedId === experiment.id
          ? 'md:col-span-2 lg:col-span-4 row-span-2 order-first'
          : 'order-none bg-ctp-base hover:shadow-md'}
            {highlighted.length > 0 && !highlighted.includes(experiment.id)
          ? 'opacity-40'
          : ''}
          "
      >
        {#if selectedId !== experiment.id}
          <ExperimentSimple bind:selectedId bind:highlighted {experiment} />
        {:else}
          <ExperimentDetailed
            bind:selectedId
            bind:highlighted
            bind:experiment={experiments[idx]}
          />
        {/if}
      </div>
    {/each}
  </div>
</section>
