<script lang="ts">
  import { Plus } from "lucide-svelte";
  import { enhance, applyAction } from "$app/forms";

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
    <h1 class="text-2xl font-semibold text-ctp-mauve">Workspaces</h1>
  </div>
  <div class="p-4">
    <form
      method="POST"
      action="/api/workspaces"
      class="flex flex-col text-ctp-text space-y-4 p-4"
      use:enhance={({ formData }) => {
        return async ({ result, update }) => {
          creating = false;
          inputData.name = "";
          inputData.description = "";
          workspaces.push(result);
          await update();
          await applyAction(result);
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
    {#each workspaces as workspace, idx}
      <div class="flex flex-col bg-ctp-surface0 p-3 rounded-md shadow">
        <h3 class="text-ctp-text font-semibold">{workspace.name}</h3>
      </div>
    {/each}
  </div>
</div>
