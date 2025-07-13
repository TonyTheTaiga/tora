<script lang="ts">
  import { closeCreateWorkspaceModal } from "$lib/state/app.svelte.js";
  import { enhance } from "$app/forms";
  import { goto } from "$app/navigation";
  import {
    BaseModal,
    ModalFormSection,
    ModalInput,
    ModalButtons,
  } from "$lib/components/modals";

  let name = $state("");
  let description = $state("");
</script>

<BaseModal title="New Workspace">
  {#snippet children()}
    <form
      method="POST"
      action="?/createWorkspace"
      class="space-y-4"
      use:enhance={() => {
        return async ({ result, update }) => {
          if (result.type === "redirect") {
            goto(result.location);
          } else if (result.type === "success") {
            await update();
            closeCreateWorkspaceModal();
          }
        };
      }}
    >
      <div class="space-y-4">
        <ModalFormSection title="workspace config">
          {#snippet children()}
            <div>
              <ModalInput
                name="name"
                placeholder="name"
                bind:value={name}
                required
              />
            </div>
            <div>
              <ModalInput
                name="description"
                type="textarea"
                rows={2}
                placeholder="description"
                bind:value={description}
                required
              />
            </div>
          {/snippet}
        </ModalFormSection>
      </div>

      <ModalButtons onCancel={closeCreateWorkspaceModal} submitText="create" />
    </form>
  {/snippet}
</BaseModal>
