<script lang="ts">
  import {
    Plus,
    LogOut,
    Trash2,
    Crown,
    Users,
    Mail,
    Check,
    X,
  } from "lucide-svelte";
  import { enhance } from "$app/forms";
  import { isWorkspace } from "$lib/types";
  import type { ApiKey } from "$lib/types";
  import WorkspaceInviteModal from "./workspace-invite-modal.svelte";
  import WorkspaceRoleBadge from "$lib/components/workspace-role-badge.svelte";

  let { data } = $props();
  let creatingWorkspace: boolean = $state(false);
  let creatingApiKey: boolean = $state(false);
  let createdKey: string = $state("");
  let workspaceError: string = $state("");
  let apiKeyError: string = $state("");
  let inviteModalOpen = $state(false);
  let workspaceToInvite: any = $state(null);
  let pendingInvitations = $state<any[]>([]);
  let invitationsLoading = $state(true);

  const ownedWorkspaces = $derived(
    data.workspaces?.filter((w) => w.role === "OWNER") || [],
  );
  const sharedWorkspaces = $derived(
    data.workspaces?.filter((w) => w.role !== "OWNER") || [],
  );

  function openInviteModal(workspace: any) {
    workspaceToInvite = workspace;
    inviteModalOpen = true;
  }

  async function sendInvitation(email: string, roleId: string) {
    if (!workspaceToInvite || !data.user) return;

    try {
      const response = await fetch("/api/workspace-invitations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workspaceId: workspaceToInvite.id,
          email,
          roleId,
        }),
      });

      if (response.ok) {
        inviteModalOpen = false;
        workspaceToInvite = null;
      }
    } catch (error) {
      console.error("Failed to send invitation:", error);
    }
  }

  async function loadInvitations() {
    try {
      const response = await fetch("/api/workspace-invitations");
      if (response.ok) {
        pendingInvitations = await response.json();
      }
    } catch (error) {
      console.error("Failed to load invitations:", error);
    } finally {
      invitationsLoading = false;
    }
  }

  async function respondToInvitation(invitationId: string, accept: boolean) {
    try {
      const response = await fetch(
        `/api/workspaces/any/invitations?invitationId=${invitationId}&action=${accept ? "accept" : "deny"}`,
        {
          method: "PATCH",
        },
      );

      if (response.ok) {
        pendingInvitations = pendingInvitations.filter(
          (inv) => inv.id !== invitationId,
        );
        if (accept) {
          window.location.reload();
        }
      }
    } catch (error) {
      console.error("Failed to respond to invitation:", error);
    }
  }

  $effect(() => {
    loadInvitations();
  });
</script>

<div class="bg-ctp-base font-mono">
  <!-- Header -->
  <div
    class="flex items-center justify-between p-6 border-b border-ctp-surface0/10"
  >
    <div class="flex items-center gap-4">
      <div class="w-2 h-8 bg-ctp-blue -full"></div>
      <div>
        <h1 class="text-xl font-bold text-ctp-text">Settings</h1>
        <div class="text-xs text-ctp-subtext0">system configuration</div>
      </div>
    </div>
  </div>

  <!-- Main content -->
  <div class="p-6 space-y-8">
    <!-- User Profile Section -->
    <div>
      <div class="text-sm text-ctp-text font-medium mb-4">user profile</div>
      
      <!-- Primary info - email as filename -->
      <div class="flex items-center gap-2 mb-3">
        <div class="text-ctp-green text-sm">●</div>
        <div class="text-sm text-ctp-text font-mono font-semibold break-words min-w-0">
          {data?.user?.email}
        </div>
        <div class="text-xs text-ctp-subtext0 font-mono ml-auto">
          {data?.user?.created_at ? new Date(data.user.created_at).toLocaleDateString("en-US", {
            month: "short",
            day: "numeric", 
            year: "2-digit",
          }) : ""}
        </div>
      </div>

      <!-- Secondary metadata -->
      <div class="pl-6 space-y-1 text-xs font-mono mb-4">
        <div class="flex items-center gap-2">
          <span class="text-ctp-subtext0 w-8">id:</span>
          <span class="text-ctp-blue truncate min-w-0">{data?.user?.id}</span>
        </div>
      </div>

      <div class="pl-6">
        <form action="/logout" method="POST">
          <button
            type="submit"
            class="bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-red hover:bg-ctp-red/10 hover:border-ctp-red/30 px-3 py-2 text-xs transition-all"
            aria-label="Sign out"
          >
            <div class="flex items-center gap-2">
              <LogOut size={12} />
              <span>logout</span>
            </div>
          </button>
        </form>
      </div>
    </div>

    <!-- Workspaces Section -->
    <div>
      <div class="text-sm text-ctp-text font-medium mb-4">workspaces</div>

      <!-- Create workspace form -->
      <div class="border border-ctp-surface0/20 p-3 mb-4">
        <form
          method="POST"
          action="?/createWorkspace"
          use:enhance
          class="space-y-3"
        >
          <div class="space-y-2">
            <div>
              <input
                id="workspace-name"
                name="name"
                placeholder="workspace_name"
                disabled={creatingWorkspace}
                class="w-full bg-ctp-surface0/20 border border-ctp-surface0/30 px-3 py-2 text-ctp-text placeholder-ctp-subtext0 focus:outline-none focus:ring-1 focus:ring-ctp-blue focus:border-ctp-blue transition-all text-sm"
                required
              />
            </div>
            <div>
              <input
                id="workspace-description"
                name="description"
                placeholder="description (optional)"
                disabled={creatingWorkspace}
                class="w-full bg-ctp-surface0/20 border border-ctp-surface0/30 px-3 py-2 text-ctp-text placeholder-ctp-subtext0 focus:outline-none focus:ring-1 focus:ring-ctp-blue focus:border-ctp-blue transition-all text-sm"
                defaultvalue=""
              />
            </div>
          </div>
          <button
            type="submit"
            disabled={creatingWorkspace}
            class="bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-blue hover:bg-ctp-blue/10 hover:border-ctp-blue/30  px-3 py-2 text-sm transition-all disabled:opacity-50"
          >
            <div class="flex items-center gap-2">
              <Plus size={14} />
              <span>{creatingWorkspace ? "creating..." : "create"}</span>
            </div>
          </button>
        </form>
      </div>

      {#if workspaceError}
        <div class="text-ctp-red text-sm mb-4">error: {workspaceError}</div>
      {/if}

      <!-- Workspace listings -->
      <div class="space-y-4">
        {#if ownedWorkspaces.length > 0}
          <div class="text-xs text-ctp-subtext0 mb-2 font-mono">owned:</div>
          <div class="space-y-1">
            {#each ownedWorkspaces as workspace}
              <div
                class="flex items-center hover:bg-ctp-surface0/10 px-1 py-1 transition-colors text-xs"
              >
                <span class="text-ctp-blue w-3">●</span>
                <span class="text-ctp-text flex-1 truncate min-w-0">{workspace.name}</span>
                <WorkspaceRoleBadge role={workspace.role} />
                <div class="flex items-center gap-1 ml-2">
                  <button
                    type="button"
                    class="text-ctp-subtext0 hover:text-ctp-blue hover:bg-ctp-surface0/30  p-1 transition-all"
                    title="Invite users"
                    onclick={() => openInviteModal(workspace)}
                  >
                    <Users size={10} />
                  </button>
                  <form method="POST" action="?/deleteWorkspace" use:enhance>
                    <input type="hidden" name="id" value={workspace.id} />
                    <button
                      type="submit"
                      class="text-ctp-subtext0 hover:text-ctp-red hover:bg-ctp-surface0/30  p-1 transition-all"
                      title="Delete workspace"
                      onclick={(e) => {
                        if (
                          !confirm(
                            "Are you sure you want to delete this workspace?",
                          )
                        ) {
                          e.preventDefault();
                        }
                      }}
                    >
                      <Trash2 size={10} />
                    </button>
                  </form>
                </div>
              </div>
            {/each}
          </div>
        {/if}

        {#if sharedWorkspaces.length > 0}
          <div class="text-xs text-ctp-subtext0 mb-2 mt-4 font-mono">
            shared:
          </div>
          <div class="space-y-1">
            {#each sharedWorkspaces as workspace}
              <div
                class="flex items-center hover:bg-ctp-surface0/10 px-1 py-1 transition-colors text-xs"
              >
                <span class="text-ctp-green w-3">●</span>
                <span class="text-ctp-text flex-1 truncate min-w-0">{workspace.name}</span>
                <WorkspaceRoleBadge role={workspace.role} />
                <div class="ml-2">
                  <form
                    method="POST"
                    action="?/removeSharedWorkspace"
                    use:enhance
                  >
                    <input type="hidden" name="userId" value={data?.user?.id} />
                    <input
                      type="hidden"
                      name="workspaceId"
                      value={workspace.id}
                    />
                    <button
                      type="submit"
                      class="text-ctp-subtext0 hover:text-ctp-red hover:bg-ctp-surface0/30  p-1 transition-all"
                      title="Leave workspace"
                      onclick={(e) => {
                        if (
                          !confirm(
                            "Are you sure you want to leave this workspace?",
                          )
                        ) {
                          e.preventDefault();
                        }
                      }}
                    >
                      <LogOut size={10} />
                    </button>
                  </form>
                </div>
              </div>
            {/each}
          </div>
        {/if}

        {#if ownedWorkspaces.length === 0 && sharedWorkspaces.length === 0}
          <div class="text-ctp-subtext0 text-sm">no workspaces found</div>
        {/if}

        {#if !invitationsLoading && pendingInvitations.length > 0}
          <div class="text-xs text-ctp-subtext0 mb-2 mt-4 font-mono">
            invitations:
          </div>
          <div class="space-y-1">
            {#each pendingInvitations as invitation}
              <div
                class="flex items-center hover:bg-ctp-surface0/10 px-1 py-1 transition-colors text-xs"
              >
                <span class="text-ctp-yellow w-3">●</span>
                <span class="text-ctp-text flex-1 truncate min-w-0">{invitation.workspaceName}</span>
                <span class="text-xs text-ctp-subtext1 truncate">from {invitation.fromEmail}</span>
                <div class="flex items-center gap-1 ml-2">
                  <button
                    type="button"
                    class="text-ctp-subtext0 hover:text-ctp-green hover:bg-ctp-surface0/30  p-1 transition-all"
                    title="Accept invitation"
                    onclick={() => respondToInvitation(invitation.id, true)}
                  >
                    <Check size={10} />
                  </button>
                  <button
                    type="button"
                    class="text-ctp-subtext0 hover:text-ctp-red hover:bg-ctp-surface0/30  p-1 transition-all"
                    title="Decline invitation"
                    onclick={() => respondToInvitation(invitation.id, false)}
                  >
                    <X size={10} />
                  </button>
                </div>
              </div>
            {/each}
          </div>
        {/if}
      </div>
    </div>

    <!-- API Keys Section -->
    <div>
      <div class="text-sm text-ctp-text font-medium mb-4">api keys</div>

      <!-- Create API key form -->
      <div class="border border-ctp-surface0/20 p-3 mb-4">
        <form
          action="?/createApiKey"
          method="POST"
          use:enhance
          class="space-y-3"
        >
          <div>
            <input
              id="key-name"
              type="text"
              name="name"
              placeholder="key_name"
              disabled={creatingApiKey}
              class="w-full bg-ctp-surface0/20 border border-ctp-surface0/30 px-3 py-2 text-ctp-text placeholder-ctp-subtext0 focus:outline-none focus:ring-1 focus:ring-ctp-blue focus:border-ctp-blue transition-all text-sm"
              required
            />
          </div>
          <button
            type="submit"
            disabled={creatingApiKey}
            class="bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-green hover:bg-ctp-green/10 hover:border-ctp-green/30  px-3 py-2 text-sm transition-all disabled:opacity-50"
          >
            <div class="flex items-center gap-2">
              <Plus size={14} />
              <span>{creatingApiKey ? "generating..." : "generate"}</span>
            </div>
          </button>
        </form>
      </div>

      {#if apiKeyError}
        <div class="text-ctp-red text-sm mb-4">error: {apiKeyError}</div>
      {/if}

      {#if createdKey !== ""}
        <div class="bg-ctp-green/10 border border-ctp-green/20 p-3 mb-4">
          <div class="text-xs text-ctp-green mb-2">
            key generated successfully:
          </div>
          <div class="bg-ctp-surface0/20 p-2 mb-2">
            <code class="text-ctp-blue text-xs break-all">{createdKey}</code>
          </div>
          <div class="text-xs text-ctp-subtext1 mb-2">
            ⚠️ save this key - it won't be shown again
          </div>
          <button
            class="bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-green hover:bg-ctp-green/10 hover:border-ctp-green/30  px-3 py-2 text-sm transition-all"
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
            class="flex items-center hover:bg-ctp-surface0/10 px-1 py-1 transition-colors text-xs"
          >
            <span class="text-{apiKey.revoked ? 'ctp-red' : 'ctp-green'} w-3">●</span>
            <span class="text-ctp-text flex-1 truncate min-w-0">{apiKey.name}</span>
            <span class="text-xs text-ctp-subtext1 w-16">{apiKey.revoked ? "revoked" : "active"}</span>
            <span class="text-xs text-ctp-subtext0 w-20 text-right truncate">{apiKey.createdAt}</span>
            {#if !apiKey.revoked}
              <div class="ml-2">
                <form method="POST" action="?/revokeApiKey" use:enhance>
                  <input type="hidden" name="id" value={apiKey.id} />
                  <button
                    type="submit"
                    class="text-ctp-subtext0 hover:text-ctp-red hover:bg-ctp-surface0/30  p-1 transition-all"
                    title="Revoke API key"
                    onclick={(e) => {
                      if (
                        !confirm(
                          "Are you sure you want to revoke this API key?",
                        )
                      ) {
                        e.preventDefault();
                      }
                    }}
                  >
                    <Trash2 size={10} />
                  </button>
                </form>
              </div>
            {/if}
          </div>
        {/each}
      </div>
    </div>
  </div>
</div>

<WorkspaceInviteModal
  bind:isOpen={inviteModalOpen}
  workspace={workspaceToInvite}
  onInvite={sendInvitation}
/>
