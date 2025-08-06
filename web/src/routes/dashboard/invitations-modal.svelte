<script lang="ts">
  import {
    BaseModal,
    ModalFormSection,
    ModalInput,
    ModalButtons,
    ModalClose,
  } from "$lib/components/modals";

  let { invitations, close } = $props();
</script>

<BaseModal title="Workspace Invitations" onClose={close}>
  {#each invitations as invitation}{/each}
</BaseModal>
