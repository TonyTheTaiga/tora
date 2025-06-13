<script lang="ts">
  import { onMount } from "svelte";
  import { Users, UserMinus, Crown, Shield, Edit, Eye, MoreVertical } from "lucide-svelte";
  import type { Workspace, WorkspaceRole } from "$lib/types";

  let { 
    workspace 
  }: { 
    workspace: Workspace; 
  } = $props();

  interface WorkspaceMember {
    id: string;
    email: string;
    role: WorkspaceRole;
    joinedAt: string;
  }

  let members = $state<WorkspaceMember[]>([]);
  let isLoading = $state(true);
  let dropdownOpen = $state<string | null>(null);

  const roleIcons = {
    OWNER: Crown,
    ADMIN: Shield,
    EDITOR: Edit,
    VIEWER: Eye
  };

  const roleColors = {
    OWNER: "bg-ctp-yellow/20 text-ctp-yellow border-ctp-yellow/30",
    ADMIN: "bg-ctp-red/20 text-ctp-red border-ctp-red/30", 
    EDITOR: "bg-ctp-blue/20 text-ctp-blue border-ctp-blue/30",
    VIEWER: "bg-ctp-green/20 text-ctp-green border-ctp-green/30"
  };

  async function loadMembers() {
    try {
      const response = await fetch(`/api/workspaces/${workspace.id}/members`);
      if (response.ok) {
        const data = await response.json();
        members = data.members || [];
      }
    } catch (error) {
      console.error("Failed to load members:", error);
    } finally {
      isLoading = false;
    }
  }

  async function removeMember(memberId: string) {
    if (!confirm("Are you sure you want to remove this member?")) return;
    
    try {
      const response = await fetch(`/api/workspaces/${workspace.id}/members/${memberId}`, {
        method: "DELETE"
      });
      
      if (response.ok) {
        members = members.filter(m => m.id !== memberId);
      }
    } catch (error) {
      console.error("Failed to remove member:", error);
    }
  }

  async function changeRole(memberId: string, newRole: WorkspaceRole) {
    try {
      const response = await fetch(`/api/workspaces/${workspace.id}/members/${memberId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role: newRole })
      });
      
      if (response.ok) {
        const memberIndex = members.findIndex(m => m.id === memberId);
        if (memberIndex !== -1) {
          members[memberIndex].role = newRole;
        }
        dropdownOpen = null;
      }
    } catch (error) {
      console.error("Failed to change role:", error);
    }
  }

  function handleClickOutside(event: MouseEvent) {
    if (event.target && (event.target as Element).closest && !(event.target as Element).closest('.member-dropdown')) {
      dropdownOpen = null;
    }
  }

  onMount(() => {
    loadMembers();
    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  });
</script>

<div class="bg-ctp-surface0/10 backdrop-blur-md rounded-2xl border border-ctp-surface0/20 p-6 shadow-xl">
  <div class="flex items-center gap-3 mb-6">
    <div class="p-2 bg-ctp-surface1/30 backdrop-blur-sm rounded-lg">
      <Users size={20} class="text-ctp-blue" />
    </div>
    <div>
      <h2 class="text-xl font-semibold text-ctp-text">Members</h2>
      <p class="text-sm text-ctp-subtext0">{workspace.name}</p>
    </div>
  </div>

  {#if isLoading}
    <div class="flex items-center justify-center py-8">
      <div class="w-6 h-6 border-2 border-ctp-blue/30 border-t-ctp-blue rounded-full animate-spin"></div>
    </div>
  {:else if members.length === 0}
    <div class="text-center py-8 text-ctp-subtext0">
      <p>No members found for this workspace.</p>
    </div>
  {:else}
    <div class="space-y-3">
      {#each members as member}
        <div class="flex items-center justify-between p-4 bg-ctp-surface0/20 backdrop-blur-sm rounded-xl border border-ctp-surface0/30 hover:border-ctp-surface0/50 transition-all">
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 bg-ctp-surface1/50 rounded-full flex items-center justify-center">
              <span class="text-sm font-medium text-ctp-text">
                {member.email?.charAt(0).toUpperCase() || '?'}
              </span>
            </div>
            <div>
              <div class="font-medium text-ctp-text">{member.email}</div>
              <div class="flex items-center gap-2">
                <span class="text-xs px-2 py-1 rounded-full border {roleColors[member.role]}">{member.role}</span>
              </div>
            </div>
          </div>

          {#if workspace.role === "OWNER" && member.role !== "OWNER"}
            <div class="relative member-dropdown">
              <button
                type="button"
                class="p-2 hover:bg-ctp-surface0/50 rounded-lg transition-all duration-200 text-ctp-subtext0 hover:text-ctp-text"
                onclick={() => dropdownOpen = dropdownOpen === member.id ? null : member.id}
              >
                <MoreVertical size={16} />
              </button>

              {#if dropdownOpen === member.id}
                <div class="absolute right-0 top-full mt-1 bg-ctp-surface0/80 backdrop-blur-xl border border-ctp-surface1/20 rounded-xl shadow-xl z-10 min-w-40">
                  <div class="p-2">
                    <div class="text-xs text-ctp-subtext0 uppercase tracking-wider px-2 py-1 mb-2">
                      Change Role
                    </div>
                    {#each (["ADMIN", "EDITOR", "VIEWER"] as WorkspaceRole[]) as role}
                      {#if role !== member.role}
                        <button
                          type="button"
                          class="w-full text-left px-3 py-2 hover:bg-ctp-surface1/30 rounded-lg transition-all duration-200 flex items-center gap-2"
                          onclick={() => changeRole(member.id, role)}
                        >
                          <span class="text-sm text-ctp-text">{role}</span>
                        </button>
                      {/if}
                    {/each}
                    <div class="border-t border-ctp-surface1/20 my-2"></div>
                    <button
                      type="button"
                      class="w-full text-left px-3 py-2 hover:bg-ctp-red/20 rounded-lg transition-all duration-200 flex items-center gap-2 text-ctp-red"
                      onclick={() => removeMember(member.id)}
                    >
                      <UserMinus size={14} />
                      <span class="text-sm">Remove</span>
                    </button>
                  </div>
                </div>
              {/if}
            </div>
          {/if}
        </div>
      {/each}
    </div>
  {/if}
</div>