<script lang="ts">
  import type { Experiment } from "$lib/types";
  import { AlertTriangle, X, Loader2 } from "lucide-svelte";
  import { onMount, onDestroy } from "svelte";

  let {
    experiment = $bindable(),
    experiments = $bindable(),
  }: {
    experiment: Experiment | null;
    experiments: Experiment[];
  } = $props();

  let isDeleting = $state(false);

  onMount(() => {
    document.body.classList.add("overflow-hidden");
  });

  onDestroy(() => {
    document.body.classList.remove("overflow-hidden");
  });

  async function deleteExperiment() {
    if (isDeleting || !experiment) return;

    isDeleting = true;

    try {
      const response = await fetch(`/api/experiments/${experiment.id}`, {
        method: "DELETE",
      });

      if (response.ok) {
        const experimentId = experiment.id;
        experiments = experiments.filter((exp) => exp.id !== experimentId);
        experiment = null;
      } else {
        console.error("Failed to delete experiment");
      }
    } catch (error) {
      console.error("Error deleting experiment:", error);
    } finally {
      isDeleting = false;
    }
  }

  function closeModal() {
    if (isDeleting) return;
    experiment = null;
  }
</script>

<div
  class="fixed inset-0 bg-ctp-crust/80 backdrop-blur-md
         flex items-center justify-center p-2 sm:p-4 z-50 overflow-hidden"
>
  <!-- MODAL CONTAINER -->
  <div
    class="bg-ctp-mantle w-full max-w-md rounded-xl border border-ctp-surface0 shadow-2xl overflow-auto max-h-[90vh]"
    role="dialog"
    aria-modal="true"
    aria-labelledby="modal-title"
  >
    <!-- HEADER -->
    <div
      class="flex items-center justify-between px-6 py-4 border-b border-ctp-surface0"
    >
      <div class="flex items-center gap-2">
        <AlertTriangle size={18} class="text-ctp-red" />
        <h2 id="modal-title" class="text-xl font-medium text-ctp-text">
          Delete Experiment?
        </h2>
      </div>
      <button
        onclick={closeModal}
        type="button"
        disabled={isDeleting}
        class="p-1.5 text-ctp-subtext0 hover:text-ctp-red hover:bg-ctp-red/10 rounded-full transition-all"
        aria-label="Close"
      >
        <X size={18} />
      </button>
    </div>

    <!-- CONTENT -->
    <div class="p-5">
      <div class="bg-ctp-red/10 border border-ctp-red/30 rounded-md p-3 mb-4">
        <p class="text-sm text-ctp-text">
          Are you sure you want to delete <strong>{experiment?.name}</strong>?
        </p>
        <p class="text-sm text-ctp-subtext0 mt-2">
          This action is permanent and cannot be undone. All experiment data,
          including metrics and hyperparameters, will be deleted.
        </p>
      </div>
    </div>

    <!-- FOOTER -->
    <div
      class="flex justify-end gap-3 pt-4 px-5 pb-5 border-t border-ctp-surface0"
    >
      <button
        onclick={closeModal}
        type="button"
        disabled={isDeleting}
        class="inline-flex items-center justify-center px-4 py-2 font-medium rounded-lg bg-transparent text-ctp-text hover:bg-ctp-surface0 transition-colors"
      >
        Cancel
      </button>
      <button
        type="button"
        onclick={deleteExperiment}
        disabled={isDeleting}
        class="inline-flex items-center justify-center gap-2 px-4 py-2 font-medium rounded-lg bg-ctp-red text-ctp-base hover:bg-ctp-red/90 hover:shadow-lg transition-all"
      >
        {#if isDeleting}
          <div class="flex items-center gap-2">
            <Loader2 size={14} class="animate-spin" />
            Deleting...
          </div>
        {:else}
          Delete
        {/if}
      </button>
    </div>
  </div>
</div>
