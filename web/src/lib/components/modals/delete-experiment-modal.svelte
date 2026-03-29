<script lang="ts">
  import type { Experiment } from "$lib/types";
  import { AlertTriangle } from "@lucide/svelte";
  import { enhance } from "$app/forms";
  import { BaseModal } from "$lib/components/modals";
  import { resetExperimentToDelete } from "$lib/state/modal.svelte.js";
  import { Button } from "$lib/components";

  let {
    experiment,
    experiments = $bindable(),
    onChange = (list: Experiment[]) => {},
  }: {
    experiment: Experiment;
    experiments: Experiment[];
    onChange?: (list: Experiment[]) => void;
  } = $props();

  let isDeleting = $state(false);

  function closeModal() {
    if (isDeleting) return;
    resetExperimentToDelete();
  }
</script>

<BaseModal
  title="Delete Experiment?"
  onClose={closeModal}
  closeDisabled={isDeleting}
>
  <div class="space-y-4">
    <div class="border border-ctp-surface0/20 p-3 mb-4">
      <div class="flex items-start gap-3">
        <AlertTriangle size={20} class="text-ctp-red mt-0.5 flex-shrink-0" />
        <div class="space-y-3 flex-1 min-w-0">
          <div
            class="bg-ctp-red/10 border border-ctp-red/30 p-3 w-full overflow-hidden"
          >
            <p class="text-sm text-ctp-text">
              Are you sure you want to delete
              <strong
                class="inline-block max-w-full truncate align-bottom"
                title={experiment?.name}>{experiment?.name}</strong
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

    <form
      method="POST"
      action="/?/deleteExperiment"
      class="flex justify-end gap-2 pt-3 mt-3 border-t border-ctp-surface0/20"
      use:enhance={() => {
        isDeleting = true;
        return async ({ result, update }) => {
          isDeleting = false;
          if (result.type === "success") {
            const experimentId = experiment.id;
            experiments = experiments.filter(
              (experiment) => experiment.id !== experimentId,
            );
            onChange?.(experiments);
            closeModal();
            await update();
          } else {
            console.error("Error deleting experiment:", result);
            closeModal();
            await update();
          }
        };
      }}
    >
      <input type="hidden" name="id" value={experiment.id} />
      <Button onclick={closeModal} type="button" disabled={isDeleting}>
        cancel
      </Button>
      <Button
        type="submit"
        variant="destructive"
        loading={isDeleting}
        loadingText="deleting..."
      >
        delete
      </Button>
    </form>
  </div>
</BaseModal>
