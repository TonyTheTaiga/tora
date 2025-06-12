<script lang="ts">
  import { Plus, LogOut, Trash2 } from "lucide-svelte";
  import { enhance } from "$app/forms";
  import { isWorkspace } from "$lib/types";
  import type { ApiKey } from "$lib/types";

  let { data } = $props();
  let creatingWorkspace: boolean = $state(false);
  let creatingApiKey: boolean = $state(false);
  let deletingWorkspace: string | null = $state(null);
  let createdKey: string = $state("");
  let workspaceError: string = $state("");
  let apiKeyError: string = $state("");
</script>

<div class="flex-1 p-2 sm:p-4 max-w-none mx-2 sm:mx-4">
  <div class="grid grid-cols-1 lg:grid-cols-3 gap-4 lg:gap-6 h-fit">
    <div
      class="bg-ctp-surface0/10 backdrop-blur-md rounded-2xl border border-ctp-surface0/20 p-4 sm:p-6 shadow-xl h-fit"
    >
      <h2 class="text-2xl font-bold text-ctp-text mb-6">User Profile</h2>
      <div class="space-y-4">
        <div class="flex flex-col space-y-2">
          <span class="text-sm font-medium text-ctp-subtext0">User ID</span>
          <div
            class="bg-ctp-surface0/20 backdrop-blur-sm rounded-lg p-3 border border-ctp-surface0/30"
          >
            <span class="text-ctp-text font-mono text-sm">{data?.user?.id}</span
            >
          </div>
        </div>
        <div class="flex flex-col space-y-2">
          <span class="text-sm font-medium text-ctp-subtext0">Email</span>
          <div
            class="bg-ctp-surface0/20 backdrop-blur-sm rounded-lg p-3 border border-ctp-surface0/30"
          >
            <span class="text-ctp-text">{data?.user?.email}</span>
          </div>
        </div>
        <div class="flex flex-col space-y-2">
          <span class="text-sm font-medium text-ctp-subtext0">Created At</span>
          <div
            class="bg-ctp-surface0/20 backdrop-blur-sm rounded-lg p-3 border border-ctp-surface0/30"
          >
            <span class="text-ctp-text">{data?.user?.created_at}</span>
          </div>
        </div>
        <div class="pt-4">
          <form action="/logout" method="POST">
            <button
              type="submit"
              class="flex items-center gap-2 px-4 py-2.5 bg-ctp-red/10 border border-ctp-red/30 rounded-lg text-ctp-red hover:bg-ctp-red hover:text-ctp-crust transition-all duration-200 font-medium backdrop-blur-sm"
              aria-label="Sign out"
            >
              <LogOut size={18} />
              <span>Sign Out</span>
            </button>
          </form>
        </div>
      </div>
    </div>

    <div
      class="bg-ctp-surface0/10 backdrop-blur-md rounded-2xl border border-ctp-surface0/20 p-4 sm:p-6 shadow-xl h-fit"
    >
      <h2 class="text-2xl font-bold text-ctp-text mb-6">Workspaces</h2>

      <form
        method="POST"
        action="?/createWorkspace"
        class="mb-6 p-4 bg-ctp-surface0/20 backdrop-blur-sm rounded-xl border border-ctp-surface0/30"
        use:enhance={() => {
          creatingWorkspace = true;
          workspaceError = "";
          return async ({ result, update }) => {
            try {
              if (result.type === "success" && result.data) {
                if (isWorkspace(result.data)) {
                  data.workspaces?.push(result.data);
                } else {
                  workspaceError = "Invalid workspace data received";
                }
              } else if (result.type === "failure") {
                workspaceError =
                  (result.data as any)?.message || "Failed to create workspace";
              } else if (result.type === "error") {
                workspaceError =
                  "An error occurred while creating the workspace";
              }
            } catch (error) {
              workspaceError = "An unexpected error occurred";
              console.error("Workspace creation error:", error);
            } finally {
              creatingWorkspace = false;
              await update();
            }
          };
        }}
      >
        <h3 class="text-lg font-semibold text-ctp-text mb-4">
          Create New Workspace
        </h3>
        <div class="grid grid-cols-1 gap-4 mb-4">
          <div class="space-y-2">
            <label
              class="text-sm font-medium text-ctp-subtext0"
              for="workspace-name"
            >
              Name
            </label>
            <input
              id="workspace-name"
              name="name"
              placeholder="Enter workspace name"
              disabled={creatingWorkspace}
              class="w-full px-4 py-3 bg-ctp-surface0/30 backdrop-blur-sm border border-ctp-surface0/40 rounded-lg text-ctp-text focus:outline-none focus:ring-2 focus:ring-ctp-mauve/50 focus:border-ctp-mauve/50 transition-all placeholder-ctp-overlay0"
              required
            />
          </div>
          <div class="space-y-2">
            <label
              class="text-sm font-medium text-ctp-subtext0"
              for="workspace-description"
            >
              Description
            </label>
            <input
              id="workspace-description"
              name="description"
              placeholder="Describe your workspace"
              disabled={creatingWorkspace}
              class="w-full px-4 py-3 bg-ctp-surface0/30 backdrop-blur-sm border border-ctp-surface0/40 rounded-lg text-ctp-text focus:outline-none focus:ring-2 focus:ring-ctp-blue/50 focus:border-ctp-blue/50 transition-all placeholder-ctp-overlay0"
            />
          </div>
        </div>
        <button
          type="submit"
          disabled={creatingWorkspace}
          class="inline-flex items-center justify-center gap-2 px-6 py-3 font-medium rounded-lg bg-ctp-blue/20 border border-ctp-blue/40 text-ctp-blue hover:bg-ctp-blue hover:text-ctp-crust transition-all duration-200 backdrop-blur-sm disabled:opacity-50"
        >
          <Plus size={18} />
          <span>{creatingWorkspace ? "Creating..." : "Create Workspace"}</span>
        </button>
      </form>

      {#if workspaceError}
        <div class="mb-4 p-3 bg-ctp-red/10 border border-ctp-red/30 rounded-lg">
          <p class="text-ctp-red text-sm">{workspaceError}</p>
        </div>
      {/if}

      <div class="space-y-4">
        {#each data.workspaces ? data.workspaces : [] as workspace}
          <div
            class="p-4 bg-ctp-surface0/20 backdrop-blur-sm rounded-xl border border-ctp-surface0/30 hover:border-ctp-surface0/50 transition-all"
          >
            <div class="flex justify-between items-start">
              <div class="flex-1">
                <h3 class="text-lg font-semibold text-ctp-text mb-2">
                  {workspace.name}
                </h3>
                <p class="text-ctp-subtext0 mb-3">
                  {workspace.description || "No description provided"}
                </p>
                <div class="text-xs text-ctp-overlay0 font-mono">
                  ID: {workspace.id}
                </div>
              </div>
              <form method="POST" action="?/deleteWorkspace" use:enhance>
                <input type="hidden" name="id" value={workspace.id} />
                <button
                  type="submit"
                  class="p-1 rounded-full text-ctp-subtext0 hover:text-ctp-red hover:bg-ctp-surface1/60 transition-colors"
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
                  <Trash2 size={14} />
                </button>
              </form>
            </div>
          </div>
        {/each}
      </div>
    </div>

    <div
      class="bg-ctp-surface0/10 backdrop-blur-md rounded-2xl border border-ctp-surface0/20 p-4 sm:p-6 shadow-xl h-fit"
    >
      <h2 class="text-2xl font-bold text-ctp-text mb-6">API Keys</h2>

      <form
        class="mb-6 p-4 bg-ctp-surface0/20 backdrop-blur-sm rounded-xl border border-ctp-surface0/30"
        use:enhance={() => {
          creatingApiKey = true;
          apiKeyError = "";

          return async ({ result, update }) => {
            try {
              if (result.type === "success" && result.data) {
                const newKey = result.data as unknown as ApiKey;
                if (newKey?.key) {
                  createdKey = newKey.key;
                } else {
                  apiKeyError = "No API key received";
                }
              } else if (result.type === "failure") {
                apiKeyError =
                  (result.data as any)?.message || "Failed to create API key";
              } else if (result.type === "error") {
                apiKeyError = "An error occurred while creating the API key";
              }
            } catch (error) {
              apiKeyError = "An unexpected error occurred";
              console.error("API key creation error:", error);
            } finally {
              creatingApiKey = false;
              await update();
            }
          };
        }}
        action="?/createApiKey"
        method="POST"
      >
        <h3 class="text-lg font-semibold text-ctp-text mb-4">
          Create New API Key
        </h3>
        <div class="grid grid-cols-1 gap-4 mb-4">
          <div class="space-y-2">
            <label class="text-sm font-medium text-ctp-subtext0" for="key-name">
              Name
            </label>
            <input
              id="key-name"
              type="text"
              name="name"
              placeholder="Enter API key name"
              disabled={creatingApiKey}
              class="w-full px-4 py-3 bg-ctp-surface0/30 backdrop-blur-sm border border-ctp-surface0/40 rounded-lg text-ctp-text focus:outline-none focus:ring-2 focus:ring-ctp-blue/50 focus:border-ctp-blue/50 transition-all placeholder-ctp-overlay0"
              required
            />
          </div>
        </div>
        <button
          type="submit"
          disabled={creatingApiKey}
          class="inline-flex items-center justify-center gap-2 px-6 py-3 font-medium rounded-lg bg-ctp-blue/20 border border-ctp-blue/40 text-ctp-blue hover:bg-ctp-blue hover:text-ctp-crust transition-all duration-200 backdrop-blur-sm disabled:opacity-50"
        >
          <Plus size={18} />
          <span>{creatingApiKey ? "Creating..." : "Create API Key"}</span>
        </button>
      </form>

      {#if apiKeyError}
        <div class="mb-4 p-3 bg-ctp-red/10 border border-ctp-red/30 rounded-lg">
          <p class="text-ctp-red text-sm">{apiKeyError}</p>
        </div>
      {/if}

      {#if createdKey !== ""}
        <div
          class="mb-6 p-4 bg-ctp-green/10 backdrop-blur-sm rounded-xl border border-ctp-green/30"
        >
          <h3 class="text-lg font-semibold text-ctp-green mb-3">
            New API Key Created
          </h3>
          <div class="bg-ctp-surface0/30 backdrop-blur-sm rounded-lg p-4 mb-4">
            <code class="text-ctp-blue font-mono text-sm break-all"
              >{createdKey}</code
            >
          </div>
          <p class="text-ctp-subtext0 text-sm mb-4">
            ⚠️ Save this key now - it won't be shown again after you copy it.
          </p>
          <button
            class="inline-flex items-center justify-center gap-2 px-6 py-3 font-medium rounded-lg bg-ctp-green/20 border border-ctp-green/40 text-ctp-green hover:bg-ctp-green hover:text-ctp-crust transition-all duration-200 backdrop-blur-sm w-full"
            type="button"
            onclick={() => {
              navigator.clipboard.writeText(createdKey);
              createdKey = "";
            }}
          >
            Copy Key
          </button>
        </div>
      {/if}

      <div class="space-y-4">
        {#each data.apiKeys ? data.apiKeys : [] as apiKey}
          <div
            class="p-4 bg-ctp-surface0/20 backdrop-blur-sm rounded-xl border border-ctp-surface0/30 hover:border-ctp-surface0/50 transition-all"
          >
            <div class="flex justify-between items-start">
              <div class="flex-1">
                <h3 class="text-lg font-semibold text-ctp-text mb-2">
                  {apiKey.name}
                </h3>
                <div class="space-y-2 text-sm">
                  <div>
                    <span class="text-ctp-subtext0">Created:</span>
                    <div class="text-ctp-text">{apiKey.createdAt}</div>
                  </div>
                  <div>
                    <span class="text-ctp-subtext0">Last Used:</span>
                    <div class="text-ctp-text">
                      {apiKey.lastUsed || "Never"}
                    </div>
                  </div>
                  <div>
                    <span class="text-ctp-subtext0">Status:</span>
                    <div class="text-ctp-text">
                      {apiKey.revoked ? "Revoked" : "Active"}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        {/each}
      </div>
    </div>
  </div>
</div>
