<script lang="ts">
  import {
    User,
    Plus,
    GalleryVertical,
    LogOut,
    KeySquare,
    Home,
  } from "lucide-svelte";
  import Toolbar from "./toolbar.svelte";
  import { enhance } from "$app/forms";
  import { isWorkspace } from "$lib/types";
  import type { ApiKey } from "$lib/types";
  import { goto } from "$app/navigation";

  let { data } = $props();
  let showUser: boolean = $state(true);
  let showWorkspaces: boolean = $state(false);
  let showApiKeys: boolean = $state(false);
  let creatingWorkspace: boolean = $state(false);
  let creatingApiKey: boolean = $state(false);
  let createdKey: string = $state("");
</script>

<Toolbar>
  <button
    onclick={() => {
      goto("/");
    }}
  >
    <Home />
  </button>
  <button
    onclick={() => {
      if (!showUser) {
        showUser = true;
        showWorkspaces = false;
        showApiKeys = false;
      }
    }}
  >
    <User />
  </button>
  <button
    onclick={() => {
      if (!showWorkspaces) {
        showWorkspaces = true;
        showUser = false;
        showApiKeys = false;
      }
    }}
  >
    <GalleryVertical />
  </button>
  <button
    onclick={() => {
      if (!showApiKeys) {
        showApiKeys = true;
        showUser = false;
        showWorkspaces = false;
      }
    }}
  >
    <KeySquare />
  </button>
</Toolbar>

<div class="text-ctp-text">
  {#if showUser}
    <div class="flex flex-col">
      <span>
        {data?.user?.id}
      </span>
      <span>
        {data?.user?.email}
      </span>
      <span>
        {data?.user?.created_at}
      </span>
      <form action="/logout" method="POST">
        <button
          type="submit"
          class="w-full sm:w-auto flex items-center gap-2 px-3 py-1.5 border border-ctp-red rounded-md text-ctp-red hover:bg-ctp-red hover:text-ctp-crust transition-colors font-medium text-sm"
          aria-label="Sign out"
        >
          <LogOut size={16} />
          <span>Sign Out</span>
        </button>
      </form>
    </div>
  {/if}

  {#if showWorkspaces}
    <div>
      <form
        method="POST"
        action="?/createWorkspace"
        class="flex flex-col gap-4"
        use:enhance={() => {
          creatingWorkspace = true;
          return async ({ result, update }) => {
            creatingWorkspace = false;
            if (result.type === "success" && result.data) {
              if (isWorkspace(result.data)) {
                data.workspaces?.push(result.data);
              } else {
                throw new Error(
                  "got back something other than a valid workspace!",
                );
              }
            }
            await update();
          };
        }}
      >
        <div class="space-y-1.5">
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
            class="w-full px-3 py-2 bg-ctp-base border-0 rounded-lg text-ctp-text focus:outline-none focus:ring-2 focus:ring-ctp-mauve transition-all placeholder-ctp-overlay0 shadow-sm"
            required
          />
        </div>
        <div class="space-y-1.5">
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
            class="w-full px-3 py-2 bg-ctp-base border-0 rounded-lg text-ctp-text focus:outline-none focus:ring-2 focus:ring-ctp-blue transition-all placeholder-ctp-overlay0 shadow-sm"
          />
        </div>
        <button
          type="submit"
          disabled={creatingWorkspace}
          class="inline-flex items-center justify-center gap-2 px-4 py-2 font-medium rounded-lg bg-ctp-blue text-ctp-crust self-start"
        >
          <Plus size={16} />
          Create Workspace
        </button>
      </form>

      {#each data.workspaces ? data.workspaces : [] as workspace}
        <div>
          <h3>{workspace.name}</h3>
          <span>{workspace.id}</span>
          <span>
            {workspace.description}
          </span>
        </div>
      {/each}
    </div>
  {/if}

  {#if showApiKeys}
    <div>
      <form
        class="mb-6"
        use:enhance={() => {
          creatingApiKey = true;

          return async ({ result, update }) => {
            await update();
            if (result.type === "success" || result.type === "failure") {
              const newKey = result.data as unknown as ApiKey;
              if (newKey.key) {
                createdKey = newKey.key;
              }
            }
            creatingApiKey = false;
          };
        }}
        action="?/createApiKey"
        method="POST"
      >
        <div class="flex flex-col sm:flex-row gap-3">
          <input
            type="text"
            name="name"
            placeholder="Key name"
            class="flex-grow bg-ctp-crust border border-ctp-surface0 rounded-md px-3 py-2 text-ctp-text placeholder-ctp-subtext0 focus:outline-none focus:border-ctp-blue focus:ring-1 focus:ring-ctp-blue/30"
            disabled={creatingApiKey}
            required
          />
          <button
            type="submit"
            class="flex items-center gap-1.5 px-3 py-1.5 border border-ctp-blue rounded-md text-ctp-blue hover:bg-ctp-blue hover:text-ctp-crust transition-colors font-medium text-sm"
          >
            <Plus size={16} />
            <span>Create</span>
          </button>
        </div>
      </form>

      {#if createdKey !== ""}
        <div>
          <p>
            <span class="text-ctp-blue">{createdKey}</span>
            Do NOT lost this key, it will be gone after you copy it.
          </p>
          <button
            class="inline-flex items-center justify-center gap-2 px-4 py-2 font-medium rounded-lg bg-ctp-blue text-ctp-crust self-start"
            type="button"
            onclick={() => {
              navigator.clipboard.writeText(createdKey);
              createdKey = "";
            }}>Copy</button
          >
        </div>
      {/if}

      {#each data.apiKeys ? data.apiKeys : [] as apiKey}
        <div class="py-2">
          <span>{apiKey.name}</span>
          <span>{apiKey.createdAt}</span>
          <span>{apiKey.lastUsed}</span>
          <span>{apiKey.revoked}</span>
        </div>
      {/each}
    </div>
  {/if}
</div>
