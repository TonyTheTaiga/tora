<script lang="ts">
  import { onMount } from "svelte";
  import {
    User,
    Calendar,
    Key,
    Plus,
    Copy,
    Trash2,
    CheckCircle,
    AlertCircle,
  } from "lucide-svelte";
  import { browser } from "$app/environment";
  import { fade } from "svelte/transition";
  import { invalidateAll } from "$app/navigation";

  const { data } = $props();

  let user = {
    id: data.user?.id || "unknown",
    username: data.user?.username || "anonymous",
    email: data.user?.email || "",
    joinedDate: new Date().toLocaleDateString(),
    apiKeys: data.user?.apiKeys || [],
  };

  let newKeyName = $state("");
  let showNewKeyValue = $state("");
  let showNewKeyModal = $state(false);
  let copied = $state(false);
  let error = $state("");

  const generateNewKey = async () => {
    if (!newKeyName.trim()) return;
    error = "";

    try {
      const response = await fetch("/api/keys", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ name: newKeyName }),
      });

      if (!response.ok) {
        throw new Error("Failed to create API key");
      }

      const data = await response.json();

      showNewKeyValue = data.key.key;
      showNewKeyModal = true;
      invalidateAll();
    } catch (err) {
      console.error("Error creating API key:", err);
      error = err instanceof Error ? err.message : "Failed to create API key";
    } finally {
      newKeyName = "";
    }
  };

  const copyToClipboard = async (text: string) => {
    if (browser) {
      await navigator.clipboard.writeText(text);
      copied = true;
      setTimeout(() => {
        copied = false;
      }, 2000);
    }
  };

  const deleteKey = async (keyId: string) => {
    error = "";
    try {
      const response = await fetch(`/api/keys?id=${keyId}`, {
        method: "DELETE",
      });

      if (!response.ok) {
        throw new Error("Failed to delete API key");
      }
      invalidateAll();
    } catch (err) {
      console.error("Error deleting API key:", err);
      error = err instanceof Error ? err.message : "Failed to delete API key";
    }
  };

  onMount(() => {
    console.log("User profile mounted:", user);
  });
</script>

<div class="w-full">
  <div class="bg-ctp-mantle rounded-lg p-6 border border-ctp-surface0">
    <div class="flex items-center gap-4 border-b border-ctp-surface0 pb-6 mb-6">
      <div class="p-3 bg-ctp-surface0 rounded-full text-ctp-lavender">
        <User size={40} />
      </div>
      <div class="flex-grow">
        <div
          class="flex flex-col md:flex-row md:items-center md:justify-between"
        >
          <div>
            <p class="text-ctp-subtext0">@{user.username}</p>
            <div
              class="flex items-center gap-1.5 mt-1 text-xs text-ctp-subtext1"
            >
              <Calendar size={14} />
              <span>Member since {user.joinedDate}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div>
      <div class="flex items-center gap-2 mb-4 text-ctp-peach">
        <Key size={20} />
        <h2 class="font-medium text-ctp-text">API Keys</h2>
      </div>

      <form
        class="mb-6"
        onsubmit={(e) => {
          e.preventDefault();
          generateNewKey();
        }}
      >
        <div class="flex flex-col sm:flex-row gap-3">
          <input
            type="text"
            bind:value={newKeyName}
            placeholder="Key name (e.g. Training Pipeline)"
            class="flex-grow bg-ctp-crust border border-ctp-surface0 rounded-md px-3 py-2 text-ctp-text placeholder-ctp-subtext0 focus:outline-none focus:border-ctp-blue"
            required
          />
          <button
            type="submit"
            class="bg-ctp-mauve hover:bg-ctp-lavender text-ctp-crust py-2 px-4 rounded-md flex items-center justify-center gap-2 transition-colors font-medium"
          >
            <Plus size={16} />
            <span>Create</span>
          </button>
        </div>

        {#if error}
          <div class="mt-3 text-ctp-red text-sm flex items-center gap-2">
            <AlertCircle size={16} />
            <span>{error}</span>
          </div>
        {/if}
      </form>

      {#if user.apiKeys.length === 0}
        <div
          class="text-ctp-subtext0 text-center py-4 border border-dashed border-ctp-surface0 rounded-md"
        >
          No API keys yet. Create one to integrate with external tools.
        </div>
      {:else}
        <div class="space-y-4">
          {#each user.apiKeys as key}
            <div class="border border-ctp-surface0 rounded-lg bg-ctp-crust p-4">
              <div class="flex justify-between items-start mb-2">
                <div>
                  <h3 class="font-medium text-ctp-text">{key.name}</h3>
                  <div class="text-xs text-ctp-subtext0 mt-1">
                    <span class="font-mono">{key.prefix}••••••••••••••••</span>
                  </div>
                </div>
                <button
                  onclick={() => deleteKey(key.id)}
                  class="text-ctp-subtext0 hover:text-ctp-red p-1.5 transition-colors"
                  aria-label="Delete key"
                >
                  <Trash2 size={16} />
                </button>
              </div>
              <div
                class="text-xs text-ctp-subtext0 flex justify-between mt-3 pt-2 border-t border-ctp-surface0"
              >
                <span
                  >Created: {new Date(key.createdAt).toLocaleDateString()}</span
                >
                {#if key.lastUsed}
                  <span
                    >Last used: {new Date(
                      key.lastUsed,
                    ).toLocaleDateString()}</span
                  >
                {:else}
                  <span>Never used</span>
                {/if}
              </div>
            </div>
          {/each}
        </div>
      {/if}
    </div>
  </div>
</div>

{#if showNewKeyModal}
  <div
    class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
    transition:fade
  >
    <div
      class="bg-ctp-base border border-ctp-surface0 rounded-lg max-w-md w-full mx-4 p-6 shadow-xl"
    >
      <div class="text-ctp-green mb-4 flex items-center gap-2">
        <CheckCircle size={24} />
        <h3 class="text-xl font-medium text-ctp-text">API Key Created</h3>
      </div>

      <p class="text-ctp-subtext0 mb-4">
        Copy your API key now. For security reasons, it won't be shown again.
      </p>

      <div
        class="bg-ctp-crust border border-ctp-surface0 rounded-md p-3 font-mono text-ctp-text mb-6 flex items-center justify-between overflow-x-auto"
      >
        <div class="truncate">{showNewKeyValue}</div>
        <button
          onclick={() => copyToClipboard(showNewKeyValue)}
          class="text-ctp-subtext0 hover:text-ctp-blue ml-2"
          aria-label="Copy to clipboard"
        >
          {#if copied}
            <CheckCircle size={18} class="text-ctp-green" />
          {:else}
            <Copy size={18} />
          {/if}
        </button>
      </div>

      <div class="flex justify-end">
        <button
          onclick={() => {
            showNewKeyModal = false;
            showNewKeyValue = "";
          }}
          class="bg-ctp-surface0 hover:bg-ctp-surface1 text-ctp-text px-4 py-2 rounded-md transition-colors font-medium"
        >
          Close
        </button>
      </div>
    </div>
  </div>
{/if}
