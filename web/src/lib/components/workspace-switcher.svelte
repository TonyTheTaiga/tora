<!-- might need to remove this component -->

<script lang="ts">
  import { ChevronDown, Plus, Briefcase } from "lucide-svelte";
  import type { Workspace } from "$lib/types";
  import { enhance } from "$app/forms";
  import { goto } from "$app/navigation";
  import { onMount } from "svelte";
  import WorkspaceRoleBadge from "./workspace-role-badge.svelte";

  let {
    currentWorkspace = $bindable(),
    workspaces = [],
  }: {
    currentWorkspace: Workspace | null;
    workspaces: Workspace[];
  } = $props();

  let dropdownRef: HTMLElement;

  const ownedWorkspaces = $derived(
    workspaces.filter((w) => w.role === "OWNER"),
  );
  const sharedWorkspaces = $derived(
    workspaces.filter((w) => w.role !== "OWNER"),
  );


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
      <span class="text-ctp-text truncate hidden sm:block max-w-32">
        {currentWorkspace?.name || "No workspace"}
      </span>
      <ChevronDown
        size={14}
        class="text-ctp-subtext0 transition-transform group-open:rotate-180 flex-shrink-0"
      />
    </summary>

    <div
      class="absolute top-full mt-2 right-0 w-64 sm:w-72 md:w-80 max-w-[calc(100vw-1rem)] bg-ctp-mantle/80 backdrop-blur-md border border-ctp-surface0/30 rounded-2xl shadow-2xl z-50 max-h-80 overflow-y-auto"
    >
      <div class="p-2 space-y-2">
        {#if workspaces.length === 0}
          <div class="text-center py-6 px-4">
            <Briefcase size={32} class="text-ctp-overlay0 mx-auto mb-3" />
            <div class="text-sm text-ctp-text mb-2">No workspaces yet</div>
            <div class="text-xs text-ctp-subtext0 mb-4">
              Create your first workspace to get started
            </div>
            <button
              type="button"
              class="px-4 py-2 bg-ctp-blue/20 hover:bg-ctp-blue/30 border border-ctp-blue/30 rounded-lg text-ctp-blue font-medium transition-all duration-200 flex items-center justify-center gap-2 w-full"
              onclick={() => {
                const detailsElement = document.getElementById('workspaceDropdown');
                detailsElement?.removeAttribute('open');
                goto('/settings');
              }}
            >
              <Plus size={14} />
              Create workspace
            </button>
          </div>
        {:else if ownedWorkspaces.length > 0}
          <div
            class="text-xs text-ctp-subtext0 uppercase tracking-wider px-2 py-1"
          >
            Your Workspaces
          </div>

          {#each ownedWorkspaces as workspace}
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
                class="w-full text-left px-3 py-2 rounded-xl hover:bg-ctp-surface0/50 transition-all flex items-start gap-2 {workspace.id ===
                currentWorkspace?.id
                  ? 'bg-ctp-surface0/70 backdrop-blur-sm'
                  : ''}"
              >
                <div class="flex-1 min-w-0">
                  <div class="flex items-center gap-2 mb-1">
                    <h4 class="text-sm font-medium text-ctp-text truncate">
                      {workspace.name}
                    </h4>
                    <WorkspaceRoleBadge role={workspace.role} />
                  </div>
                  {#if workspace.description}
                    <p class="text-xs text-ctp-subtext0 truncate">
                      {workspace.description}
                    </p>
                  {/if}
                </div>
              </button>
            </form>
          {/each}
        {/if}

        {#if sharedWorkspaces.length > 0}
          <div
            class="text-xs text-ctp-subtext0 uppercase tracking-wider px-2 py-1 {ownedWorkspaces.length >
            0
              ? 'mt-4'
              : ''}"
          >
            Shared with You
          </div>

          {#each sharedWorkspaces as workspace}
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
                class="w-full text-left px-3 py-2 rounded-xl hover:bg-ctp-surface0/50 transition-all flex items-start gap-2 {workspace.id ===
                currentWorkspace?.id
                  ? 'bg-ctp-surface0/70 backdrop-blur-sm'
                  : ''}"
              >
                <div class="flex-1 min-w-0">
                  <div class="flex items-center gap-2 mb-1">
                    <h4 class="text-sm font-medium text-ctp-text truncate">
                      {workspace.name}
                    </h4>
                    <WorkspaceRoleBadge role={workspace.role} />
                  </div>
                  {#if workspace.description}
                    <p class="text-xs text-ctp-subtext0 truncate">
                      {workspace.description}
                    </p>
                  {:else}
                    <p class="text-xs text-ctp-subtext0/70 truncate">
                      Shared workspace
                    </p>
                  {/if}
                </div>
              </button>
            </form>
          {/each}
        {/if}

        {#if workspaces.length > 0}
        <div class="border-t border-ctp-surface0 mt-2 pt-2">
          <button
            type="button"
            class="w-full text-left px-3 py-2 rounded-xl hover:bg-ctp-surface0/50 transition-all flex items-center gap-2 text-ctp-blue"
            onclick={() => {
              const detailsElement =
                document.getElementById("workspaceDropdown");
              detailsElement?.removeAttribute("open");
              goto("/settings");
            }}
          >
            <Plus size={14} />
            <span class="text-sm">Manage Workspaces</span>
          </button>
        </div>
        {/if}
      </div>
    </div>
  </details>
</div>

