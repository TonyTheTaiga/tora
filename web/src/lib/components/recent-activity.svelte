<script lang="ts">
  import type { Experiment, Workspace } from "$lib/types";

  interface Props {
    experiments: Experiment[];
    workspaces: Workspace[];
  }

  let { experiments, workspaces }: Props = $props();

  let activeTab = $state<"experiments" | "workspaces">("experiments");

  function formatDate(date: Date): string {
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year:
        date.getFullYear() !== new Date().getFullYear() ? "numeric" : undefined,
    });
  }

  function formatTime(date: Date): string {
    return date.toLocaleTimeString("en-US", {
      hour: "numeric",
      minute: "2-digit",
      hour12: true,
    });
  }
</script>

<div class="space-y-1 font-mono text-sm">
  <div class="flex gap-4 text-ctp-subtext0 mb-2">
    <button
      onclick={() => (activeTab = "experiments")}
      class="transition-colors {activeTab === 'experiments'
        ? 'text-ctp-blue'
        : 'hover:text-ctp-text'}"
    >
      [experiments]
    </button>
    <button
      onclick={() => (activeTab = "workspaces")}
      class="transition-colors {activeTab === 'workspaces'
        ? 'text-ctp-blue'
        : 'hover:text-ctp-text'}"
    >
      [workspaces]
    </button>
  </div>

  <div class="space-y-0">
    {#if activeTab === "experiments"}
      {#each experiments.slice(0, 5) as experiment}
        {#if experiment.workspaceId}
          <a
            href="/experiments/{experiment.id}"
            class="block hover:bg-ctp-surface0/20 px-1 py-1 transition-colors"
          >
            <div class="flex items-center gap-2">
              <span class="text-ctp-text truncate flex-1"
                >{experiment.name}</span
              >
              <div
                class="flex items-center gap-2 text-xs text-ctp-lavender flex-shrink-0"
              >
                <span>{formatDate(experiment.createdAt)}</span>
                <span class="hidden sm:inline text-ctp-lavender/80"
                  >{formatTime(experiment.createdAt)}</span
                >
              </div>
            </div>
          </a>
        {:else}
          <div class="hover:bg-ctp-surface0/20 px-1 py-1 transition-colors">
            <div class="flex items-center gap-2">
              <span class="text-ctp-text truncate flex-1"
                >{experiment.name}</span
              >
              <div
                class="flex items-center gap-2 text-xs text-ctp-lavender flex-shrink-0"
              >
                <span>{formatDate(experiment.createdAt)}</span>
                <span class="hidden sm:inline text-ctp-lavender/80"
                  >{formatTime(experiment.createdAt)}</span
                >
              </div>
            </div>
          </div>
        {/if}
      {/each}

      {#if experiments.length === 0}
        <div class="px-1 py-2 text-ctp-subtext0">no recent experiments</div>
      {/if}
    {:else}
      {#each workspaces.slice(0, 5) as workspace}
        <a
          href="/workspaces/{workspace.id}"
          class="block hover:bg-ctp-surface0/20 px-1 py-1 transition-colors"
        >
          <div class="flex items-center gap-2">
            <span class="text-ctp-text truncate flex-1">{workspace.name}</span>
            <div
              class="flex items-center gap-2 text-xs text-ctp-lavender flex-shrink-0"
            >
              <span>{formatDate(workspace.createdAt)}</span>
              <span class="hidden sm:inline text-ctp-lavender/80"
                >{formatTime(workspace.createdAt)}</span
              >
            </div>
          </div>
        </a>
      {/each}

      {#if workspaces.length === 0}
        <div class="px-1 py-2 text-ctp-subtext0">no workspaces found</div>
      {/if}
    {/if}
  </div>
</div>
