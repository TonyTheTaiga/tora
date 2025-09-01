<script lang="ts">
  import type { ApiKey } from "$lib/types";
  import { AlertTriangle, Loader2 } from "@lucide/svelte";
  import { enhance } from "$app/forms";
  import { BaseModal } from "$lib/components/modals";
  import { resetApiKeyToRevoke } from "$lib/state/modal.svelte.js";

  let { apiKey = $bindable() }: { apiKey: ApiKey } = $props();

  let isRevoking = $state(false);

  function closeModal() {
    if (isRevoking) return;
    resetApiKeyToRevoke();
  }
</script>

<BaseModal title="Revoke API Key?" onClose={closeModal}>
  <div class="space-y-4">
    <div class="border border-ctp-surface0/20 p-3 mb-4">
      <div class="flex items-start gap-3">
        <AlertTriangle size={20} class="text-ctp-red mt-0.5 flex-shrink-0" />
        <div class="space-y-3 flex-1 min-w-0">
          <div
            class="bg-ctp-red/10 border border-ctp-red/30 p-3 w-full overflow-hidden"
          >
            <p class="text-sm text-ctp-text">
              Are you sure you want to revoke
              <strong
                class="inline-block max-w-full truncate align-bottom"
                title={apiKey?.name}>{apiKey?.name}</strong
              >?
            </p>
            <p class="text-sm text-ctp-subtext0 mt-2">
              This action disables the key immediately. Clients using this key
              will no longer authenticate.
            </p>
          </div>
        </div>
      </div>
    </div>

    <form
      method="POST"
      action="?/revokeApiKey"
      class="flex justify-end gap-2 pt-3 mt-3 border-t border-ctp-surface0/20"
      use:enhance={() => {
        isRevoking = true;
        return async ({ result, update }) => {
          isRevoking = false;
          if (result.type === "success") {
            closeModal();
            await update();
          } else {
            console.error("Error revoking API key:", result);
            closeModal();
            await update();
          }
        };
      }}
    >
      <input type="hidden" name="keyId" value={apiKey.id} />
      <button
        onclick={closeModal}
        type="button"
        disabled={isRevoking}
        class="bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-subtext0 hover:bg-ctp-surface0/30 hover:text-ctp-text px-3 py-2 text-sm transition-all"
      >
        cancel
      </button>
      <button
        type="submit"
        disabled={isRevoking}
        class="bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-red hover:bg-ctp-red/10 hover:border-ctp-red/30 px-3 py-2 text-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {#if isRevoking}
          <div class="flex items-center gap-2">
            <Loader2 size={14} class="animate-spin" />
            revoking...
          </div>
        {:else}
          revoke
        {/if}
      </button>
    </form>
  </div>
</BaseModal>
