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

<div class="flex flex-col h-full">
  <div class="border-b border-ctp-mantle">
    <h1>Workspaces</h1>
  </div>
  <div>
    <form
      method="POST"
      action="/api/workspaces"
      class="flex flex-col text-ctp-text"
      use:enhance={({ formData }) => {
        return async ({ result, update }) => {
          creating = false;
          inputData.name = "";
          inputData.description = "";
          console.log("result", result);
          workspaces.push(result);
          await update();
          await applyAction(result);
        };
      }}
    >
      <label>
        <input
          name="name"
          bind:value={inputData.name}
          placeholder="name"
          disabled={creating}
        />
      </label>
      <label>
        <input
          name="description"
          bind:value={inputData.description}
          placeholder="Description"
          disabled={creating}
        />
      </label>
      <button type="submit" disabled={creating}>
        <div class="flex flex-row ml-auto">
          <Plus size={16}></Plus>
          <span>Create Workspace</span>
        </div>
      </button>
    </form>
  </div>

  <div class="flex-1">
    {#each workspaces as workspace, idx}
      <div class="flex flex-col">
        <h3>{workspace.name}</h3>
      </div>
    {/each}
  </div>
</div>
