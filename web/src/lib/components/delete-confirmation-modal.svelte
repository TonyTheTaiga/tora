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

  $effect(() => {
    if (experiment && typeof document !== 'undefined') {
      document.body.classList.add("overflow-hidden");
      return () => document.body.classList.remove("overflow-hidden");
    }
  });

  async function deleteExperiment() {
    if (isDeleting || !experiment) return;

    isDeleting = true;

    try {
      const response = await fetch(`/api/experiments/${experiment.id}`, {
        method: "DELETE",
      });

      if (response.ok) {
        // Remove the experiment from the experiments array
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

{#if experiment}
  <div class="fixed inset-0 z-50 flex items-center justify-center p-4">
    <!-- Backdrop -->
    <div
      class="absolute inset-0 bg-ctp-crust opacity-80"
      onclick={closeModal}
      onkeydown={(e) => e.key === "Escape" && closeModal()}
      role="presentation"
    ></div>

    <!-- Modal -->
    <div
      class="bg-ctp-base border border-ctp-surface0 rounded-lg shadow-xl max-w-md w-full z-10"
      role="dialog"
      aria-modal="true"
      aria-labelledby="modal-title"
    >
      <!-- Header -->
      <div
        class="flex items-center justify-between p-4 border-b border-ctp-surface0"
      >
        <h2
          id="modal-title"
          class="flex items-center gap-2 text-lg font-medium text-ctp-text"
        >
          <AlertTriangle size={18} class="text-ctp-red" />
          Delete Experiment?
        </h2>
        <button
          class="p-1 text-ctp-subtext0 hover:text-ctp-text rounded-md focus:outline-none focus:ring-2 focus:ring-ctp-mauve"
          onclick={closeModal}
          disabled={isDeleting}
          aria-label="Close"
        >
          <X size={18} />
        </button>
      </div>

      <!-- Content -->
      <div class="p-4">
        <div class="bg-ctp-red/10 border border-ctp-red/30 rounded-md p-3 mb-4">
          <p class="text-sm text-ctp-text">
            Are you sure you want to delete <strong>{experiment.name}</strong>?
          </p>
          <p class="text-sm text-ctp-subtext0 mt-2">
            This action is permanent and cannot be undone. All experiment data,
            including metrics and hyperparameters, will be deleted.
          </p>
        </div>
      </div>

      <!-- Footer -->
      <div class="flex justify-end gap-3 p-4 border-t border-ctp-surface0">
        <button
          type="button"
          class="px-4 py-2 text-sm bg-ctp-surface0 text-ctp-text rounded-md hover:bg-ctp-surface1 focus:outline-none focus:ring-2 focus:ring-ctp-mauve"
          onclick={closeModal}
          disabled={isDeleting}
        >
          Cancel
        </button>
        <button
          type="button"
          class="px-4 py-2 text-sm bg-ctp-red text-ctp-base rounded-md hover:bg-ctp-red/90 focus:outline-none focus:ring-2 focus:ring-ctp-red"
          onclick={deleteExperiment}
          disabled={isDeleting}
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
{/if}
