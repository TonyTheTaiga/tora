<script lang="ts">
  import { Users } from "@lucide/svelte";
  import {
    BaseModal,
    ModalFormSection,
    ModalInput,
    ModalButtons,
  } from "$lib/components/modals";

  let {
    isOpen = $bindable(false),
    workspace,
    onInvite,
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

  function handleSubmit(e: Event) {
    e.preventDefault();
    if (email && roleId) {
      onInvite(email, roleId);
      closeModal();
      email = "";
      roleId = "";
    }
  }
</script>

{#if isOpen && workspace}
  <BaseModal title="Invite User">
    {#snippet children()}
      <form onsubmit={handleSubmit} class="space-y-4">
        <ModalFormSection title="details">
          {#snippet children()}
            <div class="flex items-center gap-2 mb-3">
              <span class="text-sm text-ctp-subtext0"
                >workspace: <span class="text-ctp-blue">{workspace.name}</span
                ></span
              >
            </div>

            <div>
              <ModalInput
                id="invite-email"
                name="email"
                type="email"
                placeholder="colleague@example.com"
                bind:value={email}
                required
              />
            </div>

            <div>
              <select
                id="invite-role"
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
          {/snippet}
        </ModalFormSection>

        <ModalButtons onCancel={closeModal} submitText="send" />
      </form>
    {/snippet}
  </BaseModal>
{/if}
