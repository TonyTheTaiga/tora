<script lang="ts">
  import { onMount } from "svelte";
  import { Users, X, Send, ChevronDown } from "lucide-svelte";
  import type { Workspace, WorkspaceRole } from "$lib/types";

  let { 
    isOpen = $bindable(false), 
    workspace 
  }: { 
    isOpen: boolean; 
    workspace: Workspace; 
  } = $props();

  let email = $state("");
  let selectedRole: WorkspaceRole = $state("VIEWER");
  let isLoading = $state(false);
  let dropdownOpen = $state(false);

  const roles: { value: WorkspaceRole; label: string; description: string }[] = [
    { value: "VIEWER", label: "Viewer", description: "Can view experiments" },
    { value: "EDITOR", label: "Editor", description: "Can create and edit experiments" },
    { value: "ADMIN", label: "Admin", description: "Can manage workspace settings" }
  ];

  async function handleInvite(event: Event) {
    event.preventDefault();
    if (!email || !selectedRole) return;
    
    isLoading = true;
    try {
      const response = await fetch(`/api/workspaces/${workspace.id}/invite`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, role: selectedRole })
      });

      if (response.ok) {
        email = "";
        selectedRole = "VIEWER";
        isOpen = false;
      }
    } catch (error) {
      console.error("Failed to send invitation:", error);
    } finally {
      isLoading = false;
    }
  }

  function handleKeydown(event: KeyboardEvent) {
    if (event.key === "Escape") {
      isOpen = false;
    }
  }

  onMount(() => {
    if (typeof window !== "undefined") {
      document.addEventListener("keydown", handleKeydown);
      return () => document.removeEventListener("keydown", handleKeydown);
    }
  });
</script>

{#if isOpen}
  <div 
    class="fixed inset-0 bg-black/30 backdrop-blur-sm z-50 flex items-center justify-center p-4"
    onclick={() => isOpen = false}
  >
    <div 
      class="bg-ctp-base/80 backdrop-blur-xl border border-ctp-surface0/20 rounded-2xl shadow-2xl max-w-md w-full p-6"
      onclick={(e) => e.stopPropagation()}
    >
      <div class="flex items-center justify-between mb-6">
        <div class="flex items-center gap-3">
          <div class="p-2 bg-ctp-surface1/30 backdrop-blur-sm rounded-lg">
            <Users size={20} class="text-ctp-blue" />
          </div>
          <div>
            <h2 class="text-lg font-semibold text-ctp-text">Share Workspace</h2>
            <p class="text-sm text-ctp-subtext0">{workspace.name}</p>
          </div>
        </div>
        <button
          class="p-2 hover:bg-ctp-surface0/50 rounded-lg transition-all duration-200 text-ctp-subtext0 hover:text-ctp-text"
          onclick={() => isOpen = false}
        >
          <X size={20} />
        </button>
      </div>

      <form onsubmit={handleInvite} class="space-y-4">
        <div>
          <label class="block text-sm font-medium text-ctp-text mb-2">
            Email Address
          </label>
          <input
            type="email"
            bind:value={email}
            placeholder="colleague@example.com"
            class="w-full px-4 py-3 bg-ctp-surface0/30 backdrop-blur-sm border border-ctp-surface1/20 rounded-xl text-ctp-text placeholder-ctp-subtext0 focus:outline-none focus:ring-2 focus:ring-ctp-blue/50 focus:border-ctp-blue/30 transition-all duration-200"
            required
          />
        </div>

        <div>
          <label class="block text-sm font-medium text-ctp-text mb-2">
            Role
          </label>
          <div class="relative">
            <button
              type="button"
              class="w-full px-4 py-3 bg-ctp-surface0/30 backdrop-blur-sm border border-ctp-surface1/20 rounded-xl text-left text-ctp-text focus:outline-none focus:ring-2 focus:ring-ctp-blue/50 focus:border-ctp-blue/30 transition-all duration-200 flex items-center justify-between"
              onclick={() => dropdownOpen = !dropdownOpen}
            >
              <div>
                <div class="font-medium">{roles.find(r => r.value === selectedRole)?.label}</div>
                <div class="text-xs text-ctp-subtext0">{roles.find(r => r.value === selectedRole)?.description}</div>
              </div>
              <ChevronDown size={16} class="text-ctp-subtext0 transition-transform duration-200 {dropdownOpen ? 'rotate-180' : ''}" />
            </button>

            {#if dropdownOpen}
              <div class="absolute top-full left-0 right-0 mt-1 bg-ctp-surface0/80 backdrop-blur-xl border border-ctp-surface1/20 rounded-xl shadow-xl z-10">
                {#each roles as role}
                  <button
                    type="button"
                    class="w-full px-4 py-3 text-left hover:bg-ctp-surface1/30 transition-all duration-200 first:rounded-t-xl last:rounded-b-xl"
                    onclick={() => {
                      selectedRole = role.value;
                      dropdownOpen = false;
                    }}
                  >
                    <div class="font-medium text-ctp-text">{role.label}</div>
                    <div class="text-xs text-ctp-subtext0">{role.description}</div>
                  </button>
                {/each}
              </div>
            {/if}
          </div>
        </div>

        <div class="flex gap-3 pt-4">
          <button
            type="button"
            class="flex-1 px-4 py-3 bg-ctp-surface0/30 hover:bg-ctp-surface0/50 backdrop-blur-sm border border-ctp-surface1/20 rounded-xl text-ctp-text transition-all duration-200"
            onclick={() => isOpen = false}
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={isLoading || !email}
            class="flex-1 px-4 py-3 bg-ctp-blue/20 hover:bg-ctp-blue/30 backdrop-blur-sm border border-ctp-blue/20 rounded-xl text-ctp-blue font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {#if isLoading}
              <div class="w-4 h-4 border-2 border-ctp-blue/30 border-t-ctp-blue rounded-full animate-spin"></div>
            {:else}
              <Send size={16} />
            {/if}
            Send Invite
          </button>
        </div>
      </form>
    </div>
  </div>
{/if}