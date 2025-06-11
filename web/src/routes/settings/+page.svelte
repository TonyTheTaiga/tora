<script lang="ts">
  import { User, Plus, GalleryVertical } from "lucide-svelte";
  import Toolbar from "./toolbar.svelte";
  import { enhance } from "$app/forms";
  import { isWorkspace } from "$lib/types";

  let { data } = $props();
  let showUser: boolean = $state(true);
  let showWorkspaces: boolean = $state(false);
  let creatingWorkspace: boolean = $state(false);
</script>

<Toolbar>
  <button
    onclick={() => {
      if (!showUser) {
        showUser = true;
        showWorkspaces = false;
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
      }
    }}
  >
    <GalleryVertical />
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
          class="inline-flex items-center justify-center gap-2 px-4 py-2 font-medium rounded-lg bg-gradient-to-r from-ctp-blue to-ctp-mauve text-ctp-crust hover:shadow-lg transition-all self-start"
        >
          <Plus size={16} />
          Create Workspace
        </button>
      </form>

      {#each data.workspaces ? data.workspaces : [] as workspace}
        <h3>{workspace.name}</h3>
        <span>{workspace.id}</span>
        <span>
          {workspace.description}
        </span>
      {/each}
    </div>
  {/if}
</div>
