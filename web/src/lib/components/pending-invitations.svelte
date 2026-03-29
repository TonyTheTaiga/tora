<script lang="ts">
  import { Check, X } from "@lucide/svelte";
  import type { PendingInvitation } from "$lib/types";
  import { Button } from "$lib/components";

  interface Props {
    invitations: PendingInvitation[];
  }

  let { invitations }: Props = $props();

  function respondToInvitation(invitationId: string, accept: boolean) {
    const form = document.createElement("form");
    form.method = "POST";
    form.action = "/?/respondToInvitation";

    const invitationIdInput = document.createElement("input");
    invitationIdInput.type = "hidden";
    invitationIdInput.name = "invitationId";
    invitationIdInput.value = invitationId;
    form.appendChild(invitationIdInput);

    const actionInput = document.createElement("input");
    actionInput.type = "hidden";
    actionInput.name = "action";
    actionInput.value = accept ? "accept" : "deny";
    form.appendChild(actionInput);

    document.body.appendChild(form);
    form.submit();
  }
</script>

{#if invitations.length > 0}
  <div class="space-y-2">
    {#each invitations as invitation}
      <div
        class="flex items-center justify-between p-3 bg-ctp-surface0/10 border border-ctp-surface0/20 hover:bg-ctp-surface0/20 transition-colors"
      >
        <div class="flex-1 min-w-0">
          <div class="flex items-left gap-2">
            <span class="text-ctp-blue">name:</span>
            <span class="text-ctp-text font-medium truncate">
              {invitation.workspaceName}
            </span>
          </div>
          <div class="text-sm text-ctp-subtext1 mt-1">
            from {invitation.from}
          </div>
        </div>
        <div class="flex items-center gap-2 ml-4">
          <Button
            size="sm"
            class="text-ctp-subtext0 hover:text-ctp-green hover:border-ctp-green/30"
            title="Accept invitation"
            onclick={() => respondToInvitation(invitation.id, true)}
          >
            <Check class="w-3 h-3" />
            <span>Accept</span>
          </Button>
          <Button
            variant="destructive"
            size="sm"
            title="Decline invitation"
            onclick={() => respondToInvitation(invitation.id, false)}
          >
            <X class="w-3 h-3" />
            <span>Decline</span>
          </Button>
        </div>
      </div>
    {/each}
  </div>
{/if}
