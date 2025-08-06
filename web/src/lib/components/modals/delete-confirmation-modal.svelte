<script lang="ts">
  import type { Experiment } from "$lib/types";
  import { AlertTriangle, Loader2 } from "@lucide/svelte";
  import { resetExperimentToDelete } from "$lib/state/modal.svelte.js";
  import { enhance } from "$app/forms";
  import { BaseModal } from "$lib/components/modals";

  let {
    experiment,
    experiments = $bindable(),
  }: {
    experiment: Experiment;
    experiments: Experiment[];
  } = $props();

  let isDeleting = $state(false);

  function closeModal() {
    if (isDeleting) return;
    resetExperimentToDelete();
  }
</script>

<BaseModal title="Delete Experiment?">
  <div class="space-y-4">
    <!-- Warning content -->
    <div class="border border-ctp-surface0/20 p-3 mb-4">
      <div class="flex items-start gap-3">
        <AlertTriangle size={20} class="text-ctp-red mt-0.5 flex-shrink-0" />
        <div class="space-y-3">
          <div class="bg-ctp-red/10 border border-ctp-red/30 p-3">
            <p class="text-sm text-ctp-text">
              Are you sure you want to delete <strong>{experiment?.name}</strong
              >?
            </p>
            <p class="text-sm text-ctp-subtext0 mt-2">
              This action is permanent and cannot be undone. All experiment
              data, including metrics and hyperparameters, will be deleted.
            </p>
          </div>
        </div>
      </div>
    </div>

    <!-- Form -->
    <form
      method="POST"
      action="/experiments?/delete"
      class="flex justify-end gap-2 pt-3 mt-3 border-t border-ctp-surface0/20"
      use:enhance={() => {
        isDeleting = true;
        return async ({ result, update }) => {
          isDeleting = false;
          if (result.type === "success") {
            const experimentId = experiment.id;
            experiments = experiments.filter((exp) => exp.id !== experimentId);
            resetExperimentToDelete();
          } else {
            console.error("Error deleting experiment:", result);
            await update();
          }
        };
      }}
    >
      <input type="hidden" name="id" value={experiment.id} />
      <button
        onclick={closeModal}
        type="button"
        disabled={isDeleting}
        class="bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-subtext0 hover:bg-ctp-surface0/30 hover:text-ctp-text px-3 py-2 text-sm transition-all"
      >
        cancel
      </button>
      <button
        type="submit"
        disabled={isDeleting}
        class="bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-red hover:bg-ctp-red/10 hover:border-ctp-red/30 px-3 py-2 text-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {#if isDeleting}
          <div class="flex items-center gap-2">
            <Loader2 size={14} class="animate-spin" />
            deleting...
          </div>
        {:else}
          delete
        {/if}
      </button>
    </form>
  </div>
</BaseModal>
