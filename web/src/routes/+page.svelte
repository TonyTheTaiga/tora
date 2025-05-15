<script lang="ts">
  import CreateExperimentModal from "$lib/components/create-experiment-modal.svelte";
  import ExperimentsList from "$lib/components/experiments-list.svelte";
  import type { Experiment } from "$lib/types";
  import type { PageData } from "./$types";
  import { Plus } from "lucide-svelte";

  let { data }: { data: PageData } = $props();
  let experiments: Experiment[] = $state(data.experiments);
  let isUserSignedIn: boolean = $state(!!data.session);

  let isOpen: boolean = $state(false);
  function toggleIsOpen() {
    isOpen = !isOpen;
  }
  
  function handleExperimentButton() {
    if (isUserSignedIn) {
      isOpen = true;
    } else {
      alert("Please create an account to start logging experiments");
    }
  }
</script>

{#if isOpen}
  <CreateExperimentModal {toggleIsOpen} />
{/if}

<div class="flex flex-col space-y-2">
  <button
    onclick={handleExperimentButton}
    class="inline-flex items-center justify-center sm:gap-2 sm:px-4 sm:py-2 w-10 h-10 sm:w-auto sm:h-auto rounded-full sm:rounded-md bg-ctp-mauve text-ctp-crust hover:bg-ctp-lavender transition-colors font-medium"
    aria-label="New Experiment"
  >
    <Plus size={16} />
    <span class="hidden sm:inline">New Experiment</span></button
  >
  <ExperimentsList bind:experiments />
</div>
