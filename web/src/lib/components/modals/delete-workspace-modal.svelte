<script lang="ts">
  import type { Workspace } from "$lib/types";
  import { AlertTriangle } from "@lucide/svelte";
  import { enhance } from "$app/forms";
  import { BaseModal } from "$lib/components/modals";
  import { resetWorkspaceToDelete } from "$lib/state/modal.svelte.js";
  import { Button } from "$lib/components";

  let { workspace = $bindable() }: { workspace: Workspace } = $props();

  let isDeleting = $state(false);

  function closeModal() {
    if (isDeleting) return;
    resetWorkspaceToDelete();
  }
</script>

<BaseModal
  title="Delete Workspace?"
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
                title={workspace?.name}>{workspace?.name}</strong
              >?
            </p>
            <p class="text-sm text-ctp-subtext0 mt-2">
              This action is permanent and cannot be undone. All workspace data,
              including experiments, metrics, and member access, will be
              deleted.
            </p>
          </div>
        </div>
      </div>
    </div>

    <form
      method="POST"
      action="/?/deleteWorkspace"
      class="flex justify-end gap-2 pt-3 mt-3 border-t border-ctp-surface0/20"
      use:enhance={() => {
        isDeleting = true;
        return async ({ result, update }) => {
          isDeleting = false;
          if (result.type === "success") {
            closeModal();
            await update();
          } else {
            console.error("Error deleting workspace:", result);
            closeModal();
            await update();
          }
        };
      }}
    >
      <input type="hidden" name="workspaceId" value={workspace.id} />
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
