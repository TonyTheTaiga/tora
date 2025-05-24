<script lang="ts">
  import { ChevronDown, Plus, Briefcase } from "lucide-svelte";
  import type { Workspace } from "$lib/types";
  import { enhance } from "$app/forms";

  let {
    currentWorkspace = $bindable(),
    workspaces = [],
  }: {
    currentWorkspace: Workspace | null;
    workspaces: Workspace[];
  } = $props();
</script>

<div class="workspace-switcher relative">
  <details class="group">
    <summary
      class="flex items-center gap-2 px-2 sm:px-3 py-1.5 rounded-md border border-ctp-surface0 bg-ctp-crust hover:bg-ctp-surface0 transition-colors text-sm cursor-pointer list-none"
    >
      <Briefcase size={16} class="text-ctp-mauve flex-shrink-0" />
      <span class="text-ctp-text font-medium truncate min-w-0">
        {currentWorkspace?.name || "Select Workspace"}
      </span>
      <ChevronDown
        size={14}
        class="text-ctp-subtext0 transition-transform group-open:rotate-180 flex-shrink-0"
      />
    </summary>

    <div
      class="absolute top-full mt-1 right-0 w-64 sm:w-72 md:w-80 max-w-[calc(100vw-1rem)] bg-ctp-mantle border border-ctp-surface0 rounded-md shadow-lg z-50 max-h-80 overflow-y-auto"
    >
      <div class="p-2 space-y-2">
        <div
          class="text-xs text-ctp-subtext0 uppercase tracking-wider px-2 py-1"
        >
          Your Workspaces
        </div>

        {#each workspaces as workspace}
          <form method="POST" action="/?/switchWorkspace" use:enhance>
            <input type="hidden" name="workspaceId" value={workspace.id} />
            <button
              type="submit"
              class="w-full text-left px-2 py-1.5 rounded hover:bg-ctp-surface0 transition-colors flex items-center gap-2 {workspace.id ===
              currentWorkspace?.id
                ? 'bg-ctp-surface0'
                : ''}"
            >
              <Briefcase size={14} class="text-ctp-mauve" />
              <div class="flex-1 min-w-0">
                <div class="text-sm text-ctp-text truncate">
                  {workspace.name}
                </div>
                {#if workspace.description}
                  <div class="text-xs text-ctp-subtext0 truncate">
                    {workspace.description}
                  </div>
                {/if}
              </div>
            </button>
          </form>
        {/each}

        <div class="border-t border-ctp-surface0 mt-2 pt-2">
          <a
            href="/workspaces"
            class="w-full text-left px-2 py-1.5 rounded hover:bg-ctp-surface0 transition-colors flex items-center gap-2 text-ctp-blue"
          >
            <Plus size={14} />
            <span class="text-sm">Manage Workspaces</span>
          </a>
        </div>
      </div>
    </div>
  </details>
</div>
