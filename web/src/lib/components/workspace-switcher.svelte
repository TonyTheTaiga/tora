<!-- might need to remove this component -->

<script lang="ts">
  import { ChevronDown, Plus, Briefcase } from "lucide-svelte";
  import type { Workspace } from "$lib/types";
  import { enhance } from "$app/forms";
  import { goto } from "$app/navigation";
  import { onMount } from "svelte";

  let {
    currentWorkspace = $bindable(),
    workspaces = [],
  }: {
    currentWorkspace: Workspace | null;
    workspaces: Workspace[];
  } = $props();

  let dropdownRef: HTMLElement;

  onMount(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef && !dropdownRef.contains(event.target as Node)) {
        const detailsElement = document.getElementById("workspaceDropdown");
        detailsElement?.removeAttribute("open");
      }
    };

    document.addEventListener("click", handleClickOutside);
    return () => document.removeEventListener("click", handleClickOutside);
  });
</script>

<div class="workspace-switcher relative" bind:this={dropdownRef}>
  <details id="workspaceDropdown" class="group">
    <summary
      class="flex items-center gap-2 px-3 py-2 rounded-full border border-ctp-surface0/20 bg-ctp-crust/70 backdrop-blur-md hover:bg-ctp-surface0/50 transition-all text-sm cursor-pointer list-none shadow-sm"
    >
      <Briefcase size={16} class="text-ctp-mauve flex-shrink-0" />
      <ChevronDown
        size={14}
        class="text-ctp-subtext0 transition-transform group-open:rotate-180 flex-shrink-0"
      />
    </summary>

    <div
      class="absolute top-full mt-2 right-0 w-64 sm:w-72 md:w-80 max-w-[calc(100vw-1rem)] bg-ctp-mantle/80 backdrop-blur-md border border-ctp-surface0/30 rounded-2xl shadow-2xl z-50 max-h-80 overflow-y-auto"
    >
      <div class="p-2 space-y-2">
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
              return async ({ result, update }) => {
                await update();

                const detailsElement =
                  document.getElementById("workspaceDropdown");
                detailsElement?.removeAttribute("open");
              };
            }}
          >
            <input type="hidden" name="workspaceId" value={workspace.id} />
            <button
              type="submit"
              class="w-full text-left px-3 py-2 rounded-xl hover:bg-ctp-surface0/50 transition-all flex items-center gap-2 {workspace.id ===
              currentWorkspace?.id
                ? 'bg-ctp-surface0/70 backdrop-blur-sm'
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
          <button
            type="button"
            class="w-full text-left px-3 py-2 rounded-xl hover:bg-ctp-surface0/50 transition-all flex items-center gap-2 text-ctp-blue"
            onclick={() => {
              const detailsElement =
                document.getElementById("workspaceDropdown");
              detailsElement?.removeAttribute("open");
              goto("/workspaces");
            }}
          >
            <Plus size={14} />
            <span class="text-sm">Manage Workspaces</span>
          </button>
        </div>
      </div>
    </div>
  </details>
</div>
