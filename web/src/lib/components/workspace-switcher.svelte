<script lang="ts">
  import { ChevronDown, Plus, Briefcase } from "lucide-svelte";
  import { goto } from "$app/navigation";
  import { enhance } from "$app/forms";
  import type { Workspace } from "$lib/types";
  import { browser } from "$app/environment";

  let {
    currentWorkspace = $bindable(),
    workspaces = [],
  }: {
    currentWorkspace: Workspace | null;
    workspaces: Workspace[];
  } = $props();

  let isOpen = $state(false);
  let switchingWorkspace = $state(false);
  let buttonElement: HTMLButtonElement;
  let dropdownPosition = $state({ top: 0, left: 0, width: 0 });

  function updateDropdownPosition() {
    if (buttonElement && browser) {
      const rect = buttonElement.getBoundingClientRect();
      dropdownPosition = {
        top: rect.bottom + 4,
        left: rect.right - 256, // 256px = w-64
        width: rect.width
      };
    }
  }

  function handleClickOutside(event: MouseEvent) {
    const target = event.target as HTMLElement;
    if (!target.closest(".workspace-switcher")) {
      isOpen = false;
    }
  }

  function toggleDropdown() {
    isOpen = !isOpen;
    if (isOpen) {
      updateDropdownPosition();
    }
  }

  $effect(() => {
    if (isOpen) {
      document.addEventListener("click", handleClickOutside);
      window.addEventListener("scroll", updateDropdownPosition);
      window.addEventListener("resize", updateDropdownPosition);
      return () => {
        document.removeEventListener("click", handleClickOutside);
        window.removeEventListener("scroll", updateDropdownPosition);
        window.removeEventListener("resize", updateDropdownPosition);
      };
    }
  });
</script>

<div class="workspace-switcher relative">
  <button
    bind:this={buttonElement}
    onclick={toggleDropdown}
    class="flex items-center gap-2 px-3 py-1.5 rounded-md border border-ctp-surface0 bg-ctp-crust hover:bg-ctp-surface0 transition-colors text-sm"
  >
    <Briefcase size={16} class="text-ctp-mauve" />
    <span class="text-ctp-text font-medium">
      {currentWorkspace?.name || "Select Workspace"}
    </span>
    <ChevronDown size={14} class="text-ctp-subtext0 transition-transform {isOpen ? 'rotate-180' : ''}" />
  </button>
</div>

<!-- Portal dropdown to document body to prevent layout issues -->
{#if isOpen && browser}
  <div
    class="fixed w-64 bg-ctp-mantle border border-ctp-surface0 rounded-md shadow-lg z-50 max-h-80 overflow-y-auto"
    style="top: {dropdownPosition.top}px; left: {dropdownPosition.left}px;"
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
          onclick={() => {
            isOpen = false;
            goto("/workspaces");
          }}
          class="w-full text-left px-2 py-1.5 rounded hover:bg-ctp-surface0 transition-colors flex items-center gap-2 text-ctp-blue"
        >
          <Plus size={14} />
          <span class="text-sm">Manage Workspaces</span>
        </button>
      </div>
    </div>
  </div>
{/if}
