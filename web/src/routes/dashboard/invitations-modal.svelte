<script lang="ts">
  import { BaseModal } from "$lib/components/modals";
  import PendingInvitations from "$lib/components/pending-invitations.svelte";

  let { invitations, close } = $props();
</script>

<BaseModal title="Workspace Invitations" onClose={close}>
  <PendingInvitations {invitations} />
</BaseModal>
