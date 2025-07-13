<script lang="ts">
  import { Check, X } from "@lucide/svelte";

  interface Invitation {
    id: string;
    workspaceId: string;
    from: string;
  }

  interface Props {
    invitations: Invitation[];
  }

  let { invitations }: Props = $props();

  function respondToInvitation(invitationId: string, accept: boolean) {
    const form = document.createElement("form");
    form.method = "POST";
    form.action = "?/respondToInvitation";

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
  <div class="section-divider" data-label="pending invitations"></div>
  <div class="surface-accent-lavender layer-spacing-md stack-layer">
    <div class="space-y-2">
      {#each invitations as invitation}
        <div
          class="flex items-center justify-between p-3 bg-ctp-surface0/10 border border-ctp-surface0/20 hover:bg-ctp-surface0/20 transition-colors"
        >
          <div class="flex-1 min-w-0">
            <div class="flex items-center gap-2">
              <span class="text-ctp-yellow">●</span>
              <span class="text-ctp-text font-medium truncate">
                {invitation.workspaceId}
              </span>
            </div>
            <div class="text-sm text-ctp-subtext1 mt-1">
              from {invitation.from}
            </div>
          </div>
          <div class="flex items-center gap-2 ml-4">
            <button
              type="button"
              class="flex items-center gap-1 text-xs text-ctp-subtext0 hover:text-ctp-green transition-colors bg-ctp-surface0/20 backdrop-blur-md border border-ctp-surface0/30 hover:border-ctp-green/30 p-2"
              title="Accept invitation"
              onclick={() => respondToInvitation(invitation.id, true)}
            >
              <Check class="w-3 h-3" />
              <span>Accept</span>
            </button>
            <button
              type="button"
              class="flex items-center gap-1 text-xs text-ctp-subtext0 hover:text-ctp-red transition-colors bg-ctp-surface0/20 backdrop-blur-md border border-ctp-surface0/30 hover:border-ctp-red/30 p-2"
              title="Decline invitation"
              onclick={() => respondToInvitation(invitation.id, false)}
            >
              <X class="w-3 h-3" />
              <span>Decline</span>
            </button>
          </div>
        </div>
      {/each}
    </div>
  </div>
{/if}
