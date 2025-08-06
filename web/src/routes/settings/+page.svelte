<script lang="ts">
  import { Plus, LogOut, Trash2 } from "@lucide/svelte";
  import { enhance } from "$app/forms";

  let { data } = $props();
  let createdKey: string = $state("");

  function revokeApiKey(keyId: string) {
    if (!confirm("Are you sure you want to revoke this API key?")) return;

    const form = document.createElement("form");
    form.method = "POST";
    form.action = "?/revokeApiKey";

    const keyIdInput = document.createElement("input");
    keyIdInput.type = "hidden";
    keyIdInput.name = "keyId";
    keyIdInput.value = keyId;
    form.appendChild(keyIdInput);

    document.body.appendChild(form);
    form.submit();
  }
</script>

<div class="font-mono">
  <!-- Header -->
  <div
    class="flex items-center justify-between border-b border-ctp-surface0/20 px-6 py-4"
  >
    <div>
      <h1 class="text-lg font-medium text-ctp-text">Settings</h1>
      <p class="text-sm text-ctp-subtext1">
        {data?.user?.email || "system configuration"}
      </p>
    </div>
    <form action="/logout" method="POST">
      <button
        type="submit"
        class="bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-red hover:bg-ctp-red/10 hover:border-ctp-red/30 px-3 py-2 text-sm transition-all"
        aria-label="Sign out"
      >
        <div class="flex items-center gap-2">
          <LogOut size={12} />
          <span>logout</span>
        </div>
      </button>
    </form>
  </div>

  <!-- Main content -->
  <div class="px-6 py-6 space-y-8">
    <!-- API Keys Section -->
    <div>
      <div class="text-base text-ctp-text font-medium mb-4">api keys</div>

      <!-- Create API key form -->
      <div class="border border-ctp-surface0/20 p-3 mb-4">
        <form
          method="POST"
          action="?/createApiKey"
          use:enhance={() => {
            return async ({ result, update }) => {
              if (result.type === "success" && result.data?.key) {
                createdKey = result.data.key as string;
              }
              await update({ reset: true });
            };
          }}
          class="space-y-3"
        >
          <div class="flex gap-2">
            <input
              id="key-name"
              type="text"
              name="name"
              placeholder="key_name"
              class="flex-1 bg-ctp-surface0/20 border border-ctp-surface0/30 px-3 py-2 text-ctp-text placeholder-ctp-subtext0 focus:outline-none focus:ring-1 focus:ring-ctp-blue focus:border-ctp-blue transition-all text-sm"
              required
            />
            <button
              type="submit"
              class="bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-green hover:bg-ctp-green/10 hover:border-ctp-green/30 px-3 py-2 text-sm transition-all disabled:opacity-50"
            >
              <div class="flex items-center gap-2">
                <Plus size={14} />
              </div>
            </button>
          </div>
        </form>
      </div>

      {#if createdKey !== ""}
        <div class="bg-ctp-green/10 border border-ctp-green/20 p-3 mb-4">
          <div class="text-sm text-ctp-green mb-2">
            key generated successfully:
          </div>
          <div class="bg-ctp-surface0/20 p-2 mb-2">
            <code class="text-ctp-blue text-sm break-all">{createdKey}</code>
          </div>
          <div class="text-sm text-ctp-subtext1 mb-2">
            ⚠️ save this key - it won't be shown again
          </div>
          <button
            class="bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-green hover:bg-ctp-green/10 hover:border-ctp-green/30 px-3 py-2 text-sm transition-all"
            type="button"
            onclick={() => {
              navigator.clipboard.writeText(createdKey);
              createdKey = "";
            }}
          >
            copy & close
          </button>
        </div>
      {/if}

      <!-- API Keys listings -->
      <div class="space-y-1">
        {#each data.apiKeys ? data.apiKeys : [] as apiKey}
          <div
            class="flex items-center hover:bg-ctp-surface0/10 px-1 py-1 transition-colors text-sm"
          >
            <span class="text-{apiKey.revoked ? 'ctp-red' : 'ctp-green'} w-3"
            ></span>
            <span class="text-ctp-text flex-1 truncate min-w-0"
              >{apiKey.name}</span
            >
            <span class="text-sm text-ctp-subtext1 w-16"
              >{apiKey.revoked ? "revoked" : "active"}</span
            >
            <span class="text-sm text-ctp-subtext0 w-20 text-right truncate"
              >{apiKey.createdAt}</span
            >
            {#if !apiKey.revoked}
              <div class="ml-2">
                <button
                  type="button"
                  class="text-ctp-subtext0 hover:text-ctp-red hover:bg-ctp-surface0/30 p-1 transition-all"
                  title="Revoke API key"
                  onclick={() => revokeApiKey(apiKey.id)}
                >
                  <Trash2 size={10} />
                </button>
              </div>
            {/if}
          </div>
        {/each}
      </div>
    </div>
  </div>
</div>
