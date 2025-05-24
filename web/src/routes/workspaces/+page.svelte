<script lang="ts">
  import { Plus, ArrowLeft, Briefcase } from "lucide-svelte";
  import { enhance, applyAction } from "$app/forms";
  import { goto, invalidateAll } from "$app/navigation";

  let { data } = $props();
  let workspaces = $state(data.workspaces);
  let creating = $state<boolean>(false);
  let inputData = $state({
    name: "",
    description: "",
  });
</script>

<div class="flex flex-col h-full bg-ctp-base text-ctp-text">
  <div class="border-b border-ctp-surface0 p-4">
    <div class="flex items-center justify-between">
      <h1 class="text-2xl font-semibold text-ctp-mauve">Workspaces</h1>
      <button
        onclick={() => goto("/")}
        class="flex items-center gap-2 text-ctp-subtext0 hover:text-ctp-text transition-colors"
      >
        <ArrowLeft size={16} />
        <span class="text-sm">Back to experiments</span>
      </button>
    </div>
  </div>
  <div class="p-4">
    <form
      method="POST"
      action="/api/workspaces"
      class="flex flex-col text-ctp-text space-y-4 p-4"
      use:enhance={({ formData }) => {
        creating = true;
        return async ({ result, update }) => {
          creating = false;
          inputData.name = "";
          inputData.description = "";
          workspaces.push(result);
          await update();
        };
      }}
    >
      <label class="text-ctp-subtext1 text-sm font-medium block mb-1">
        Name
        <input
          name="name"
          bind:value={inputData.name}
          placeholder="name"
          disabled={creating}
          class="bg-ctp-surface0 border border-ctp-surface2 text-ctp-text placeholder-ctp-subtext0 p-2 rounded-md focus:border-ctp-blue focus:ring-1 focus:ring-ctp-blue w-full"
        />
      </label>
      <label class="text-ctp-subtext1 text-sm font-medium block mb-1">
        Description
        <input
          name="description"
          bind:value={inputData.description}
          placeholder="Description"
          disabled={creating}
          class="bg-ctp-surface0 border border-ctp-surface2 text-ctp-text placeholder-ctp-subtext0 p-2 rounded-md focus:border-ctp-blue focus:ring-1 focus:ring-ctp-blue w-full"
        />
      </label>
      <button
        type="submit"
        disabled={creating}
        class="bg-ctp-blue text-ctp-base px-4 py-2 rounded-md hover:bg-ctp-sapphire focus:outline-none focus:ring-2 focus:ring-ctp-blue focus:ring-opacity-50 flex items-center justify-center space-x-2 self-start"
      >
        <div class="flex flex-row items-center space-x-2">
          <Plus size={16}></Plus>
          <span>Create Workspace</span>
        </div>
      </button>
    </form>
  </div>

  <div class="flex-1 p-4 space-y-3">
    {#if workspaces.length === 0}
      <div class="text-center py-8">
        <Briefcase size={48} class="mx-auto text-ctp-surface2 mb-4" />
        <p class="text-ctp-subtext0">
          No workspaces yet. Create your first workspace above!
        </p>
      </div>
    {:else}
      {#each workspaces as workspace}
        <form method="POST" action="/?/switchWorkspace" use:enhance>
          <input type="hidden" name="workspaceId" value={workspace.id} />
          <button
            type="submit"
            class="w-full flex flex-col bg-ctp-surface0 p-4 rounded-md shadow hover:bg-ctp-surface1 transition-colors cursor-pointer text-left"
          >
            <div class="flex items-start justify-between">
              <div class="flex-1">
                <h3 class="text-ctp-text font-semibold flex items-center gap-2">
                  <Briefcase size={16} class="text-ctp-mauve" />
                  {workspace.name}
                </h3>
                {#if workspace.description}
                  <p class="text-ctp-subtext0 text-sm mt-1">
                    {workspace.description}
                  </p>
                {/if}
              </div>
              {#if data.currentWorkspace?.id === workspace.id}
                <span
                  class="text-xs bg-ctp-blue text-ctp-base px-2 py-1 rounded"
                >
                  Current
                </span>
              {/if}
            </div>
          </button>
        </form>
      {/each}
    {/if}
  </div>
</div>
