<script lang="ts">
  import CreateExperimentModal from "$lib/components/create-experiment-modal.svelte";
  import ExperimentsList from "$lib/components/experiments-list.svelte";
  import type { Experiment } from "$lib/types";
  import type { PageData } from "./$types";
  import Toolbar from "$lib/components/toolbar.svelte";
  import DeleteConfirmationModal from "$lib/components/delete-confirmation-modal.svelte";
  import EditExperimentModal from "$lib/components/edit-experiment-modal.svelte";
  import { page } from "$app/state";
  import { UserRound } from "lucide-svelte";

  let { data }: { data: PageData } = $props();
  let experiments: Experiment[] = $state(data.experiments);
  let hasExperiments: boolean = $derived(experiments.length > 0);

  let modalState = $state({
    createExperiment: false,
    selectedForDelete: null as Experiment | null,
    selectedForEdit: null as Experiment | null,
    selectedExperiment: null as Experiment | null,
  });
</script>

{#if page.data.user}
  {#if modalState.createExperiment}
    <CreateExperimentModal
      bind:createNewExperimentFlag={modalState.createExperiment}
    />
  {/if}

  {#if modalState.selectedForDelete}
    <DeleteConfirmationModal
      bind:experiment={modalState.selectedForDelete}
      bind:experiments
    />
  {/if}

  {#if modalState.selectedForEdit}
    <EditExperimentModal bind:experiment={modalState.selectedForEdit} />
  {/if}

  <Toolbar
    bind:selectedExperiment={modalState.selectedExperiment}
    bind:isOpenCreate={modalState.createExperiment}
    {hasExperiments}
  />

  <div>
    {experiments.length} Experiments
  </div>

  <ExperimentsList
    bind:experiments
    bind:createNewExperimentFlag={modalState.createExperiment}
    bind:selectedForEdit={modalState.selectedForEdit}
    bind:selectedForDelete={modalState.selectedForDelete}
    bind:selectedExperiment={modalState.selectedExperiment}
  ></ExperimentsList>
{:else}
  <div class="flex items-center justify-center h-full">
    <div
      class="rounded-lg bg-ctp-crust border-2 border-dashed border-ctp-subtext0 p-4"
    >
      <article class="flex flex-col h-42 w-48">
        <div class="pb-2">
          <div
            class="w-8 h-8 flex items-center justify-center rounded-full border border-dashed border-ctp-blue text-ctp-blue"
          >
            <UserRound size={16} />
          </div>
        </div>

        <!-- Content -->
        <div class="flex-grow flex flex-col">
          <h3 class="text-sm font-medium text-ctp-subtext0 mb-1">
            Create an Account
          </h3>
          <p class="text-xs text-ctp-subtext0 leading-relaxed">
            Sign up to start tracking your experiments and metrics or browse
            public experiments.
          </p>
        </div>

        <!-- Footer -->
        <div class="pt-auto">
          <a
            href="/auth"
            class="px-2.5 py-1 rounded-md border border-ctp-blue text-ctp-blue hover:bg-ctp-blue hover:text-ctp-crust transition-colors font-medium text-xs"
          >
            Sign Up / Login
          </a>
        </div>
      </article>
    </div>
  </div>
{/if}
