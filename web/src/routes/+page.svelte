<script lang="ts">
  import CreateExperimentModal from "$lib/components/create-experiment-modal.svelte";
  import ExperimentsList from "$lib/components/experiments-list.svelte";
  import NewExperimentCard from "$lib/components/new-experiment-card.svelte";
  import type { Experiment } from "$lib/types";
  import type { PageData } from "./$types";

  let { data }: { data: PageData } = $props();
  let experiments: Experiment[] = $state(data.experiments);
  let isUserSignedIn: boolean = $state(!!data.session);

  let isOpen: boolean = $state(false);
  function toggleIsOpen() {
    isOpen = !isOpen;
  }
  
  function openModal() {
    isOpen = true;
  }
</script>

{#if isOpen}
  <CreateExperimentModal {toggleIsOpen} />
{/if}

<div class="flex flex-col space-y-4">
  <ExperimentsList bind:experiments>
    <NewExperimentCard 
      bind:isUserSignedIn 
      openModal={openModal} 
      slot="prepend"
    />
  </ExperimentsList>
</div>
