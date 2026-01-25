<script lang="ts">
  import {
    Plus,
    LogOut,
    Trash2,
    Key,
    Copy,
    Check,
    MoreHorizontal,
  } from "@lucide/svelte";
  import { enhance } from "$app/forms";
  import { RevokeApiKeyModal } from "$lib/components/modals";
  import { DropdownMenu } from "bits-ui";
  import {
    setApiKeyToRevoke,
    getApiKeyToRevoke,
  } from "$lib/state/modal.svelte.js";

  let { data } = $props();
  let createdKey: string = $state("");
  let apiKeyToRevoke = $derived(getApiKeyToRevoke());
  let copied = $state(false);

  function copyKey() {
    navigator.clipboard.writeText(createdKey);
    copied = true;
    setTimeout(() => {
      copied = false;
      createdKey = "";
    }, 1500);
  }
</script>

<div
  class="min-h-0 min-w-0 grow grid overflow-hidden p-4 gap-2"
  style="grid-template-columns: 240px 1fr;"
>
  <!-- Sidebar -->
  <section
    class="bg-ctp-surface0/12 shadow-lg shadow-ctp-crust/20 backdrop-blur-sm min-h-0 overflow-y-auto flex flex-col"
  >
    <div
      class="sticky top-0 z-10 surface-elevated border-b border-ctp-surface0/30 p-4"
    >
      <div class="flex items-center justify-between">
        <h2 class="text-ctp-text font-medium text-base">Settings</h2>
        <DropdownMenu.Root>
          <DropdownMenu.Trigger
            class="menu-trigger floating-element p-2 rounded-none"
          >
            <MoreHorizontal size={16} />
          </DropdownMenu.Trigger>
          <DropdownMenu.Portal>
            <DropdownMenu.Content align="end" class="menu-content text-sm">
              <DropdownMenu.Group>
                <form action="/logout" method="POST">
                  <DropdownMenu.Item
                    class="menu-item flex items-center gap-2 text-ctp-red"
                    asChild
                  >
                    <button
                      type="submit"
                      class="w-full flex items-center gap-2"
                    >
                      <LogOut size={14} />
                      <span>Sign out</span>
                    </button>
                  </DropdownMenu.Item>
                </form>
              </DropdownMenu.Group>
            </DropdownMenu.Content>
          </DropdownMenu.Portal>
        </DropdownMenu.Root>
      </div>
      <div class="text-[11px] font-mono text-ctp-overlay0 mt-1 truncate">
        {data?.user?.email || ""}
      </div>
    </div>

    <nav class="p-2">
      <div
        class="flex items-center gap-2 px-3 py-2 bg-ctp-surface0/30 text-ctp-text text-sm"
      >
        <Key size={14} />
        <span>API Keys</span>
      </div>
    </nav>
  </section>

  <!-- Main Content -->
  <section
    class="bg-ctp-surface0/18 shadow-lg shadow-ctp-crust/20 backdrop-blur-sm min-h-0 overflow-y-auto flex flex-col"
  >
    <div
      class="sticky top-0 z-10 surface-elevated border-b border-ctp-surface0/30 p-4"
    >
      <div class="flex items-center justify-between mb-2">
        <h2 class="text-ctp-text font-medium text-base">API Keys</h2>
      </div>

      <!-- Create form in header -->
      <form
        method="POST"
        action="?/createApiKey"
        use:enhance={() => {
          return async ({ result, update }) => {
            if (result.type === "success" && result.data?.key) {
              createdKey = result.data.key as string;
            }
            await update();
          };
        }}
        class="flex items-center gap-2"
      >
        <div
          class="flex-1 flex items-center bg-ctp-surface0/30 border border-ctp-surface0/40 focus-within:ring-1 focus-within:ring-ctp-blue/30 transition-all"
        >
          <input
            type="text"
            name="name"
            placeholder="key name..."
            class="flex-1 bg-transparent border-0 py-2 px-3 text-ctp-text placeholder-ctp-subtext0 focus:outline-none text-sm"
            required
          />
        </div>
        <button
          type="submit"
          class="floating-element p-2 rounded-none"
          title="Create key"
        >
          <Plus size={16} />
        </button>
      </form>
    </div>

    <div class="p-4 flex-1">
      <!-- New Key Banner -->
      {#if createdKey !== ""}
        <div class="bg-ctp-green/5 border border-ctp-green/20 p-3 mb-4">
          <div class="flex items-center justify-between mb-2">
            <span class="text-xs text-ctp-green font-medium"
              >new key created</span
            >
            <button
              type="button"
              onclick={copyKey}
              class="floating-element p-1.5 rounded-none"
              title="Copy and dismiss"
            >
              {#if copied}
                <Check size={14} class="text-ctp-green" />
              {:else}
                <Copy size={14} />
              {/if}
            </button>
          </div>
          <code
            class="text-[11px] font-mono text-ctp-blue break-all select-all block"
            >{createdKey}</code
          >
          <div class="text-[10px] text-ctp-overlay0 mt-2">
            copy now — won't be shown again
          </div>
        </div>
      {/if}

      <!-- Keys List -->
      {#if data.apiKeys && data.apiKeys.length > 0}
        <div class="space-y-1">
          {#each data.apiKeys as apiKey}
            <div
              class="flex items-center gap-3 px-2 py-2 hover:bg-ctp-surface0/20 transition-colors group"
            >
              <Key
                size={12}
                class={apiKey.revoked ? "text-ctp-overlay0" : "text-ctp-green"}
              />
              <span
                class="flex-1 text-sm text-ctp-text truncate {apiKey.revoked
                  ? 'line-through opacity-50'
                  : ''}">{apiKey.name}</span
              >
              <span class="text-[10px] font-mono text-ctp-overlay0"
                >{apiKey.createdAt}</span
              >
              {#if !apiKey.revoked}
                <button
                  type="button"
                  class="opacity-0 group-hover:opacity-100 p-1 text-ctp-overlay0 hover:text-ctp-red transition-all"
                  title="Revoke"
                  onclick={() => setApiKeyToRevoke(apiKey)}
                >
                  <Trash2 size={12} />
                </button>
              {:else}
                <span class="text-[10px] text-ctp-red">revoked</span>
              {/if}
            </div>
          {/each}
        </div>
      {:else}
        <div class="text-center py-8 text-ctp-subtext0 text-sm">
          no api keys
        </div>
      {/if}
    </div>
  </section>
</div>

{#if apiKeyToRevoke}
  <RevokeApiKeyModal bind:apiKey={apiKeyToRevoke} />
{/if}

<style>
  :global(.menu-content) {
    background: var(--color-ctp-base);
    border: 1px solid var(--color-ctp-surface0);
    backdrop-filter: blur(12px);
    color: var(--color-ctp-text);
    min-width: 10rem;
    outline: none;
    padding: 0.25rem;
    border-radius: 0;
    box-shadow:
      0 10px 15px -3px rgb(0 0 0 / 0.1),
      0 4px 6px -4px rgb(0 0 0 / 0.1);
    z-index: 50;
  }

  :global(.menu-item) {
    color: var(--color-ctp-text);
    padding: 0.5rem 0.625rem;
    transition:
      background-color 0.25s ease,
      color 0.25s ease;
    border: none;
    outline: none;
    border-radius: 0;
    cursor: pointer;
  }

  :global(.menu-item:hover) {
    background: var(--color-ctp-surface0);
  }

  :global(.menu-trigger) {
    background: transparent;
    border: none;
    outline: none;
    cursor: pointer;
  }

  :global(.menu-trigger:focus),
  :global(.menu-trigger:focus-visible) {
    outline: none;
    box-shadow: none;
  }
</style>
