<script lang="ts">
  import { ChevronDown, Plus, Briefcase } from "lucide-svelte";
  import { goto } from "$app/navigation";
  import { enhance } from "$app/forms";
  import type { Workspace } from "$lib/types";

  let {
    currentWorkspace = $bindable(),
    workspaces = [],
  }: {
    currentWorkspace: Workspace | null;
    workspaces: Workspace[];
  } = $props();

  let isOpen = $state(false);
  let switchingWorkspace = $state(false);

  function handleClickOutside(event: MouseEvent) {
    const target = event.target as HTMLElement;
    if (!target.closest(".workspace-switcher")) {
      isOpen = false;
    }
  }

  $effect(() => {
    if (isOpen) {
      document.addEventListener("click", handleClickOutside);
      return () => document.removeEventListener("click", handleClickOutside);
    }
  });
</script>

<div class="workspace-switcher relative">
  <button
    onclick={() => (isOpen = !isOpen)}
    class="flex items-center gap-2 px-3 py-1.5 rounded-md border border-ctp-surface0 bg-ctp-crust hover:bg-ctp-surface0 transition-colors text-sm"
  >
    <Briefcase size={16} class="text-ctp-mauve" />
    <span class="text-ctp-text font-medium">
      {currentWorkspace?.name || "Select Workspace"}
    </span>
    <ChevronDown size={14} class="text-ctp-subtext0" />
  </button>

  {#if isOpen}
    <div
      class="absolute top-full mt-1 left-0 w-64 bg-ctp-mantle border border-ctp-surface0 rounded-md shadow-lg z-50"
    >
      <div class="p-2">
        <div
          class="text-xs text-ctp-subtext0 uppercase tracking-wider px-2 py-1"
        >
          Your Workspaces
        </div>

        {#each workspaces as workspace}
          <form
            method="POST"
            action="/?/switchWorkspace"
            use:enhance={() => {
              switchingWorkspace = true;
              isOpen = false;
              return async ({ update }) => {
                await update();
                switchingWorkspace = false;
              };
            }}
          >
            <input type="hidden" name="workspaceId" value={workspace.id} />
            <button
              type="submit"
              disabled={switchingWorkspace}
              class="w-full text-left px-2 py-1.5 rounded hover:bg-ctp-surface0 transition-colors flex items-center gap-2 {workspace.id ===
              currentWorkspace?.id
                ? 'bg-ctp-surface0'
                : ''}"
            >
              <Briefcase size={14} class="text-ctp-mauve" />
              <div class="flex-1">
                <div class="text-sm text-ctp-text">{workspace.name}</div>
                {#if workspace.description}
                  <div class="text-xs text-ctp-subtext0">
                    {workspace.description}
                  </div>
                {/if}
              </div>
            </button>
          </form>
        {/each}

        <div class="border-t border-ctp-surface0 mt-2 pt-2">
          <button
            onclick={() => goto("/workspaces")}
            class="w-full text-left px-2 py-1.5 rounded hover:bg-ctp-surface0 transition-colors flex items-center gap-2 text-ctp-blue"
          >
            <Plus size={14} />
            <span class="text-sm">Manage Workspaces</span>
          </button>
        </div>
      </div>
    </div>
  {/if}
</div>
