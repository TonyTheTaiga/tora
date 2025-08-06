<script lang="ts">
  import { BaseModal } from "$lib/components/modals";
  import PendingInvitations from "$lib/components/pending-invitations.svelte";

  let { invitations, close } = $props();
</script>

<BaseModal title="Workspace Invitations" onClose={close}>
  {#if invitations.length > 0}
    <PendingInvitations {invitations} />
  {:else}
    <div class="text-center text-ctp-gray-400">No pending invitations.</div>
  {/if}
</BaseModal>
