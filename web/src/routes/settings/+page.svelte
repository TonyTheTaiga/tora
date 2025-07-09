<script lang="ts">
  import { Plus, LogOut, Trash2, Users, Check, X } from "@lucide/svelte";
  import WorkspaceInviteModal from "./workspace-invite-modal.svelte";
  import WorkspaceRoleBadge from "$lib/components/workspace-role-badge.svelte";
  import { enhance } from "$app/forms";

  let { data } = $props();

  let createdKey: string = $state("");
  let inviteModalOpen = $state(false);
  let workspaceToInvite: any = $state(null);

  const ownedWorkspaces = $derived(
    data.workspaces?.filter((w: any) => w.role === "OWNER") || [],
  );
  const sharedWorkspaces = $derived(
    data.workspaces?.filter((w: any) => w.role !== "OWNER") || [],
  );
  let pendingInvitations = $derived(data.invitations ? data.invitations : []);

  function openInviteModal(workspace: any) {
    workspaceToInvite = workspace;
    inviteModalOpen = true;
  }

  function sendInvitation(email: string, roleId: string) {
    if (!workspaceToInvite || !data.user) return;

    // Create and submit form programmatically
    const form = document.createElement("form");
    form.method = "POST";
    form.action = "?/sendInvitation";

    const workspaceIdInput = document.createElement("input");
    workspaceIdInput.type = "hidden";
    workspaceIdInput.name = "workspaceId";
    workspaceIdInput.value = workspaceToInvite.id;
    form.appendChild(workspaceIdInput);

    const emailInput = document.createElement("input");
    emailInput.type = "hidden";
    emailInput.name = "email";
    emailInput.value = email;
    form.appendChild(emailInput);

    const roleIdInput = document.createElement("input");
    roleIdInput.type = "hidden";
    roleIdInput.name = "roleId";
    roleIdInput.value = roleId;
    form.appendChild(roleIdInput);

    document.body.appendChild(form);
    form.submit();

    inviteModalOpen = false;
    workspaceToInvite = null;
  }

  function deleteWorkspace(workspaceId: string) {
    if (!confirm("Are you sure you want to delete this workspace?")) return;

    const form = document.createElement("form");
    form.method = "POST";
    form.action = "?/deleteWorkspace";

    const workspaceIdInput = document.createElement("input");
    workspaceIdInput.type = "hidden";
    workspaceIdInput.name = "workspaceId";
    workspaceIdInput.value = workspaceId;
    form.appendChild(workspaceIdInput);

    document.body.appendChild(form);
    form.submit();
  }

  function leaveWorkspace(workspaceId: string) {
    if (!confirm("Are you sure you want to leave this workspace?")) return;

    const form = document.createElement("form");
    form.method = "POST";
    form.action = "?/leaveWorkspace";

    const workspaceIdInput = document.createElement("input");
    workspaceIdInput.type = "hidden";
    workspaceIdInput.name = "workspaceId";
    workspaceIdInput.value = workspaceId;
    form.appendChild(workspaceIdInput);

    document.body.appendChild(form);
    form.submit();
  }

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

<div class="font-mono">
  <!-- Header -->
  <div
    class="flex items-center justify-between p-4 md:p-6 border-b border-ctp-surface0/10"
  >
    <div
      class="flex items-stretch gap-3 md:gap-4 min-w-0 flex-1 pr-4 min-h-fit"
    >
      <div
        class="w-2 bg-ctp-blue rounded-full flex-shrink-0 self-stretch"
      ></div>
      <div class="min-w-0 flex-1 py-1">
        <h1 class="text-lg md:text-xl text-ctp-text truncate font-mono">
          Settings
        </h1>
        <div class="text-sm text-ctp-subtext0 space-y-1">
          <div>system configuration</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Main content -->
  <div class="px-4 md:px-6 py-6 space-y-8">
    <!-- User Profile Section -->
    <div>
      <div class="text-base text-ctp-text font-medium mb-4">user profile</div>

      <!-- Primary info - email as filename -->
      <div class="flex items-center gap-2 mb-3">
        <div class="text-ctp-green text-sm"></div>
        <div
          class="text-sm text-ctp-text font-mono font-semibold break-words min-w-0"
        >
          {data?.user?.email}
        </div>
        <div class="text-sm text-ctp-subtext0 font-mono ml-auto">
          {data?.user?.created_at
            ? new Date(data.user.created_at).toLocaleDateString("en-US", {
                month: "short",
                day: "numeric",
                year: "2-digit",
              })
            : ""}
        </div>
      </div>

      <!-- Secondary metadata -->
      <div class="pl-6 space-y-1 text-sm font-mono mb-4">
        <div class="flex items-center gap-2">
          <span class="text-ctp-subtext0 w-8">id:</span>
          <span class="text-ctp-blue truncate min-w-0">{data?.user?.id}</span>
        </div>
      </div>

      <div class="pl-6">
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
    </div>

    <!-- Workspaces Section -->
    <div>
      <div class="text-base text-ctp-text font-medium mb-4">workspaces</div>

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
                class="w-full bg-ctp-surface0/20 border border-ctp-surface0/30 px-3 py-2 text-ctp-text placeholder-ctp-subtext0 focus:outline-none focus:ring-1 focus:ring-ctp-blue focus:border-ctp-blue transition-all text-sm"
                required
              />
            </div>
            <div class="flex gap-2">
              <input
                id="workspace-description"
                name="description"
                placeholder="description (optional)"
                class="flex-1 bg-ctp-surface0/20 border border-ctp-surface0/30 px-3 py-2 text-ctp-text placeholder-ctp-subtext0 focus:outline-none focus:ring-1 focus:ring-ctp-blue focus:border-ctp-blue transition-all text-sm"
                defaultvalue=""
              />
              <button
                type="submit"
                class="bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-blue hover:bg-ctp-blue/10 hover:border-ctp-blue/30 px-3 py-2 text-sm transition-all disabled:opacity-50"
              >
                <div class="flex items-center gap-2">
                  <Plus size={14} />
                </div>
              </button>
            </div>
          </div>
        </form>
      </div>

      <!-- Workspace listings -->
      <div class="space-y-4">
        {#if ownedWorkspaces.length > 0}
          <div class="text-sm text-ctp-subtext0 mb-2 font-mono">owned:</div>
          <div class="space-y-1">
            {#each ownedWorkspaces as workspace}
              <div
                class="flex items-center hover:bg-ctp-surface0/10 px-1 py-1 transition-colors text-sm"
              >
                <span class="text-ctp-blue w-3"></span>
                <a
                  href="/workspaces/{workspace.id}"
                  class="text-ctp-text hover:text-ctp-blue flex-1 truncate min-w-0 transition-colors"
                >
                  {workspace.name}
                </a>
                <WorkspaceRoleBadge role={workspace.role} />
                <div class="flex items-center gap-1 ml-2">
                  <button
                    type="button"
                    class="text-ctp-subtext0 hover:text-ctp-blue hover:bg-ctp-surface0/30 p-1 transition-all"
                    title="Invite users"
                    onclick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      openInviteModal(workspace);
                    }}
                  >
                    <Users size={10} />
                  </button>
                  <button
                    type="button"
                    class="text-ctp-subtext0 hover:text-ctp-red hover:bg-ctp-surface0/30 p-1 transition-all"
                    title="Delete workspace"
                    onclick={(e) => {
                      e.stopPropagation();
                      deleteWorkspace(workspace.id);
                    }}
                  >
                    <Trash2 size={10} />
                  </button>
                </div>
              </div>
            {/each}
          </div>
        {/if}

        {#if sharedWorkspaces.length > 0}
          <div class="text-sm text-ctp-subtext0 mb-2 mt-4 font-mono">
            shared:
          </div>
          <div class="space-y-1">
            {#each sharedWorkspaces as workspace}
              <div
                class="flex items-center hover:bg-ctp-surface0/10 px-1 py-1 transition-colors text-sm"
              >
                <span class="text-ctp-green w-3"></span>
                <a
                  href="/workspaces/{workspace.id}"
                  class="text-ctp-text hover:text-ctp-blue flex-1 truncate min-w-0 transition-colors"
                >
                  {workspace.name}
                </a>
                <WorkspaceRoleBadge role={workspace.role} />
                <div class="ml-2">
                  <button
                    type="button"
                    class="text-ctp-subtext0 hover:text-ctp-red hover:bg-ctp-surface0/30 p-1 transition-all"
                    title="Leave workspace"
                    onclick={(e) => {
                      e.stopPropagation();
                      leaveWorkspace(workspace.id);
                    }}
                  >
                    <LogOut size={10} />
                  </button>
                </div>
              </div>
            {/each}
          </div>
        {/if}

        {#if ownedWorkspaces.length === 0 && sharedWorkspaces.length === 0}
          <div class="text-ctp-subtext0 text-base">no workspaces found</div>
        {/if}

        {#if pendingInvitations.length > 0}
          <div class="text-sm text-ctp-subtext0 mb-2 mt-4 font-mono">
            invitations:
          </div>
          <div class="space-y-1">
            {#each pendingInvitations as invitation}
              <div
                class="flex items-center hover:bg-ctp-surface0/10 px-1 py-1 transition-colors text-sm"
              >
                <span class="text-ctp-yellow w-3"></span>
                <span class="text-ctp-text flex-1 truncate min-w-0"
                  >{invitation.workspaceId}</span
                >
                <span class="text-sm text-ctp-subtext1 truncate"
                  >from {invitation.from}</span
                >
                <div class="flex items-center gap-1 ml-2">
                  <button
                    type="button"
                    class="text-ctp-subtext0 hover:text-ctp-green hover:bg-ctp-surface0/30 p-1 transition-all"
                    title="Accept invitation"
                    onclick={() => respondToInvitation(invitation.id, true)}
                  >
                    <Check size={10} />
                  </button>
                  <button
                    type="button"
                    class="text-ctp-subtext0 hover:text-ctp-red hover:bg-ctp-surface0/30 p-1 transition-all"
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

<WorkspaceInviteModal
  bind:isOpen={inviteModalOpen}
  workspace={workspaceToInvite}
  workspaceRoles={data.workspaceRoles || []}
  onInvite={sendInvitation}
/>
