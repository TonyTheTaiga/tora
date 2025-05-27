<script lang="ts">
  import {
    Plus,
    ArrowLeft,
    Briefcase,
    Copy,
    ClipboardCheck,
  } from "lucide-svelte";
  import { enhance, applyAction } from "$app/forms";
  import { goto, invalidateAll } from "$app/navigation";

  let { data } = $props();
  let workspaces = $state(data.workspaces);
  let creating = $state<boolean>(false);
  let copiedWorkspaceId = $state<string | null>(null);
  let inputData = $state({
    name: "",
    description: "",
  });

  function copyWorkspaceId(workspaceId: string) {
    navigator.clipboard.writeText(workspaceId);
    copiedWorkspaceId = workspaceId;
    setTimeout(() => {
      copiedWorkspaceId = null;
    }, 800);
  }
</script>

<div class="flex flex-col h-full bg-ctp-base text-ctp-text">
  <div class="border-b border-ctp-surface0 p-4">
    <div class="flex items-center justify-between">
      <h1 class="text-2xl font-semibold text-ctp-text">Workspaces</h1>
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
    <div class="bg-ctp-mantle rounded-xl border border-ctp-surface0 shadow-lg">
      <div class="px-6 py-4 border-b border-ctp-surface0">
        <h3 class="text-lg font-medium text-ctp-text">Create New Workspace</h3>
      </div>
      <form
        method="POST"
        action="/api/workspaces"
        class="flex flex-col gap-4 p-5"
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
            bind:value={inputData.name}
            placeholder="Enter workspace name"
            disabled={creating}
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
            bind:value={inputData.description}
            placeholder="Describe your workspace"
            disabled={creating}
            class="w-full px-3 py-2 bg-ctp-base border-0 rounded-lg text-ctp-text focus:outline-none focus:ring-2 focus:ring-ctp-blue transition-all placeholder-ctp-overlay0 shadow-sm"
          />
        </div>
        <button
          type="submit"
          disabled={creating}
          class="inline-flex items-center justify-center gap-2 px-4 py-2 font-medium rounded-lg bg-gradient-to-r from-ctp-blue to-ctp-mauve text-ctp-crust hover:shadow-lg transition-all self-start"
        >
          <Plus size={16} />
          Create Workspace
        </button>
      </form>
    </div>
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
        {#if data.currentWorkspace?.id === workspace.id}
          <div
            class="bg-ctp-mantle rounded-xl border border-ctp-surface0 shadow-lg"
          >
            <div class="flex flex-col p-5">
              <div class="flex items-start justify-between">
                <div class="flex-1">
                  <h3
                    class="text-ctp-text font-semibold flex items-center gap-2"
                  >
                    <Briefcase size={16} class="text-ctp-mauve" />
                    {workspace.name}
                  </h3>
                  <div class="flex items-center gap-2 mt-1">
                    <span class="text-ctp-subtext1 text-xs">ID:</span>
                    <button
                      type="button"
                      class="text-xs font-mono transition-all duration-150 flex items-center gap-1 hover:bg-ctp-surface0 px-2 py-1 rounded"
                      class:text-ctp-green={copiedWorkspaceId === workspace.id}
                      class:text-ctp-subtext1={copiedWorkspaceId !==
                        workspace.id}
                      onclick={() => copyWorkspaceId(workspace.id)}
                      title="Click to copy workspace ID"
                    >
                      {#if copiedWorkspaceId === workspace.id}
                        <ClipboardCheck size={12} class="animate-bounce" />
                        Copied!
                      {:else}
                        {workspace.id}
                        <Copy size={10} class="opacity-50" />
                      {/if}
                    </button>
                  </div>
                  {#if workspace.description}
                    <p class="text-ctp-subtext0 text-sm mt-2">
                      {workspace.description}
                    </p>
                  {/if}
                </div>
                <div class="flex items-center gap-2">
                  <span
                    class="text-xs bg-ctp-blue text-ctp-base px-2 py-1 rounded font-medium"
                  >
                    Current
                  </span>
                </div>
              </div>
            </div>
          </div>
        {:else}
          <form method="POST" action="/?/switchWorkspace" use:enhance>
            <input type="hidden" name="workspaceId" value={workspace.id} />
            <button
              type="submit"
              class="w-full bg-ctp-mantle rounded-xl border border-ctp-surface0 shadow-lg hover:shadow-xl hover:bg-ctp-surface0 transition-all"
            >
              <div class="flex flex-col p-5 text-left">
                <div class="flex items-start justify-between">
                  <div class="flex-1">
                    <h3
                      class="text-ctp-text font-semibold flex items-center gap-2"
                    >
                      <Briefcase size={16} class="text-ctp-mauve" />
                      {workspace.name}
                    </h3>
                    <div class="flex items-center gap-2 mt-1">
                      <span class="text-ctp-subtext1 text-xs">ID:</span>
                      <span
                        class="text-xs font-mono text-ctp-subtext1 px-2 py-1 rounded"
                      >
                        {workspace.id}
                      </span>
                    </div>
                    {#if workspace.description}
                      <p class="text-ctp-subtext0 text-sm mt-2">
                        {workspace.description}
                      </p>
                    {/if}
                  </div>
                </div>
              </div>
            </button>
          </form>
        {/if}
      {/each}
    {/if}
  </div>
</div>
