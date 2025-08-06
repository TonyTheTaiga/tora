<script lang="ts">
  import { enhance } from "$app/forms";
  import {
    BaseModal,
    ModalFormSection,
    ModalInput,
    ModalButtons,
  } from "$lib/components/modals";

  let {
    isOpen = $bindable(false),
    workspace,
    workspaceRoles = [],
  }: {
    isOpen: boolean;
    workspace: any;
    onInvite: (email: string, roleId: string) => void;
    workspaceRoles: Array<{ id: string; name: string }>;
  } = $props();

  let email = $state("");
  let roleId = $state("");

  function closeModal() {
    isOpen = false;
  }
</script>

{#if isOpen && workspace}
  <BaseModal title="Invite User">
    <form
      method="POST"
      action="/?/sendInvitation"
      class="space-y-4"
      use:enhance={({ formElement, formData, action, cancel, submitter }) => {
        return async ({ result, update }) => {
          await update();
          email = "";
          roleId = "";
          closeModal();
        };
      }}
    >
      <input
        type="hidden"
        id="workspaceId"
        name="workspaceId"
        value={workspace.id}
      />
      <ModalFormSection title="details">
        <div class="flex items-center gap-2 mb-3">
          <span class="text-sm text-ctp-subtext0"
            >workspace: <span class="text-ctp-blue">{workspace.name}</span
            ></span
          >
        </div>

        <div>
          <ModalInput
            id="email"
            name="email"
            type="email"
            placeholder="colleague@example.com"
            bind:value={email}
            required
          />
        </div>

        <div>
          <select
            id="roleId"
            name="roleId"
            bind:value={roleId}
            required
            class="w-full bg-ctp-surface0/20 border border-ctp-surface0/30 px-3 py-2 text-ctp-text placeholder-ctp-subtext0 focus:outline-none focus:ring-1 focus:ring-ctp-blue focus:border-ctp-blue transition-all text-sm"
          >
            <option value="">select role...</option>
            {#each workspaceRoles as role}
              <option value={role.id}>{role.name}</option>
            {/each}
          </select>
        </div>
      </ModalFormSection>

      <ModalButtons onCancel={closeModal} submitText="send" />
    </form>
  </BaseModal>
{/if}
