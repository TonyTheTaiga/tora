<script lang="ts">
  import { Users } from "lucide-svelte";

  let {
    isOpen = $bindable(false),
    workspace,
    onInvite,
  }: {
    isOpen: boolean;
    workspace: any;
    onInvite: (email: string, roleId: string) => Promise<void>;
  } = $props();

  let workspaceRoles = $state<Array<{ id: string; name: string }>>([]);
  let loading = $state(false);

  async function loadRoles() {
    if (!isOpen) return;

    loading = true;
    try {
      const rolesRes = await fetch("/api/workspace-roles");

      if (rolesRes.ok) {
        workspaceRoles = await rolesRes.json();
      }
    } catch (error) {
      console.error("Failed to load roles:", error);
    } finally {
      loading = false;
    }
  }

  $effect(() => {
    if (isOpen) {
      loadRoles();
    }
  });

  function closeModal() {
    isOpen = false;
  }

  async function handleSubmit(e: Event) {
    e.preventDefault();
    const formData = new FormData(e.target as HTMLFormElement);
    const email = formData.get("email");
    const roleId = formData.get("roleId");

    if (email && roleId) {
      await onInvite(email.toString(), roleId.toString());
      closeModal();
    }
  }
</script>

{#if isOpen && workspace}
  <div
    class="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4"
  >
    <div
      class="bg-ctp-mantle/95 backdrop-blur-md border border-ctp-surface0/30 rounded-2xl shadow-2xl w-full max-w-md p-6"
    >
      <div class="flex items-center gap-3 mb-4">
        <Users size={24} class="text-ctp-blue" />
        <h3 class="text-xl font-bold text-ctp-text">
          Invite User to {workspace.name}
        </h3>
      </div>

      {#if loading}
        <div class="flex items-center justify-center py-8">
          <div
            class="w-6 h-6 border-2 border-ctp-blue/30 border-t-ctp-blue rounded-full animate-spin"
          ></div>
        </div>
      {:else}
        <form onsubmit={handleSubmit}>
          <div class="space-y-4">
            <div>
              <label class="text-sm font-medium text-ctp-subtext0 block mb-2"
                >Email Address</label
              >
              <input
                type="email"
                name="email"
                required
                placeholder="colleague@example.com"
                class="w-full px-4 py-3 bg-ctp-surface0/30 backdrop-blur-sm border border-ctp-surface0/40 rounded-lg text-ctp-text focus:outline-none focus:ring-2 focus:ring-ctp-blue/50 focus:border-ctp-blue/50 transition-all placeholder-ctp-overlay0"
              />
            </div>

            <div>
              <label class="text-sm font-medium text-ctp-subtext0 block mb-2"
                >Role</label
              >
              <select
                name="roleId"
                required
                class="w-full px-4 py-3 bg-ctp-surface0/30 backdrop-blur-sm border border-ctp-surface0/40 rounded-lg text-ctp-text focus:outline-none focus:ring-2 focus:ring-ctp-blue/50 focus:border-ctp-blue/50 transition-all"
              >
                <option value="">Select a role...</option>
                {#each workspaceRoles as role}
                  <option value={role.id}>{role.name}</option>
                {/each}
              </select>
            </div>
          </div>

          <div class="flex gap-3 mt-6">
            <button
              type="button"
              class="flex-1 px-4 py-2 border border-ctp-surface0/40 rounded-lg text-ctp-subtext0 hover:bg-ctp-surface0/20 transition-colors"
              onclick={closeModal}
            >
              Cancel
            </button>
            <button
              type="submit"
              class="flex-1 px-4 py-2 bg-ctp-blue/20 border border-ctp-blue/40 rounded-lg text-ctp-blue hover:bg-ctp-blue hover:text-ctp-crust transition-all"
            >
              Send Invitation
            </button>
          </div>
        </form>
      {/if}
    </div>
  </div>
{/if}
