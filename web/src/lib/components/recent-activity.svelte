<script lang="ts">
  import type { Experiment, Workspace } from "$lib/types";

  interface Props {
    experiments: Experiment[];
    workspaces: Workspace[];
  }

  let { experiments, workspaces }: Props = $props();

  let activeTab = $state<"experiments" | "workspaces">("experiments");

  function formatDate(date: Date): string {
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffDays === 0) {
      return "Today";
    } else if (diffDays === 1) {
      return "Yesterday";
    } else if (diffDays < 7) {
      return `${diffDays} days ago`;
    } else {
      return date.toLocaleDateString();
    }
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
              <span class="text-ctp-lavender text-sm">
                {formatDate(experiment.createdAt)}
              </span>
            </div>
          </a>
        {:else}
          <div class="hover:bg-ctp-surface0/20 px-1 py-1 transition-colors">
            <div class="flex items-center gap-2">
              <span class="text-ctp-text truncate flex-1"
                >{experiment.name}</span
              >
              <span class="text-ctp-lavender text-sm">
                {formatDate(experiment.createdAt)}
              </span>
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
            <span class="text-ctp-lavender text-sm">
              {formatDate(workspace.createdAt)}
            </span>
          </div>
        </a>
      {/each}

      {#if workspaces.length === 0}
        <div class="px-1 py-2 text-ctp-subtext0">no workspaces found</div>
      {/if}
    {/if}
  </div>
</div>
