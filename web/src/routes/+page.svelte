<script lang="ts">
  import CreateExperimentModal from "$lib/components/create-experiment-modal.svelte";
  import ExperimentsList from "$lib/components/experiments-list.svelte";
  import type { Experiment } from "$lib/types";
  import type { PageData } from "./$types";
  import Toolbar from "$lib/components/toolbar.svelte";

  let { data }: { data: PageData } = $props();
  let experiments: Experiment[] = $state(data.experiments);
  let isOpen: boolean = $state(false);
</script>

{#if isOpen}
  <CreateExperimentModal bind:isOpen />
{/if}

<Toolbar />
<ExperimentsList bind:experiments bind:isOpen></ExperimentsList>
