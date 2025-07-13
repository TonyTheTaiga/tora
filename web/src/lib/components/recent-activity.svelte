<script lang="ts">
  import type { Experiment, Workspace } from "$lib/types";
  import { Activity, FolderOpen } from "@lucide/svelte";

  interface Props {
    experiments: Experiment[];
    workspaces: Workspace[];
  }

  let { experiments, workspaces }: Props = $props();

  let timelineItems = $derived.by(() => {
    const items: Array<{
      id: string;
      name: string;
      createdAt: Date;
      type: "experiment" | "workspace";
      workspaceId?: string;
    }> = [];

    experiments.forEach((exp) => {
      items.push({
        id: exp.id,
        name: exp.name,
        createdAt: exp.createdAt,
        type: "experiment",
        workspaceId: exp.workspaceId,
      });
    });

    workspaces.forEach((ws) => {
      items.push({
        id: ws.id,
        name: ws.name,
        createdAt: ws.createdAt,
        type: "workspace",
      });
    });

    return items
      .sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime())
      .slice(0, 8);
  });

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
  {#if timelineItems.length === 0}
    <div class="px-1 py-2 text-ctp-subtext0">no recent activity</div>
  {:else}
    {#each timelineItems as item}
      {@const href =
        item.type === "experiment"
          ? `/experiments/${item.id}`
          : `/workspaces/${item.id}`}
      {@const canNavigate =
        item.type === "workspace" ||
        (item.type === "experiment" && item.workspaceId)}

      {#if canNavigate}
        <a
          {href}
          class="block hover:bg-ctp-surface0/20 px-1 py-1 transition-colors group"
        >
          <div
            class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1 sm:gap-2"
          >
            <div class="flex items-center gap-2 min-w-0 flex-1">
              <div
                class="inline-flex items-center gap-1.5 px-2 py-1 min-w-0 max-w-full {item.type ===
                'experiment'
                  ? 'bg-ctp-blue/10 text-ctp-blue border border-ctp-blue/20'
                  : 'bg-ctp-mauve/10 text-ctp-mauve border border-ctp-mauve/20'}"
              >
                <div class="flex-shrink-0">
                  {#if item.type === "experiment"}
                    <Activity size={12} />
                  {:else}
                    <FolderOpen size={12} />
                  {/if}
                </div>
                <span
                  class="text-ctp-text group-hover:text-ctp-blue transition-colors truncate min-w-0"
                  title={item.name}
                >
                  {item.name}
                </span>
              </div>
            </div>

            <!-- Timestamp -->
            <div
              class="flex items-center gap-1 text-xs text-ctp-lavender flex-shrink-0 sm:ml-2"
            >
              <span>{formatDate(item.createdAt)}</span>
              <span class="hidden sm:inline text-ctp-lavender/80">
                {formatTime(item.createdAt)}
              </span>
            </div>
          </div>
        </a>
      {:else}
        <div class="px-1 py-1">
          <div
            class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1 sm:gap-2"
          >
            <div class="flex items-center gap-2 min-w-0 flex-1">
              <div
                class="inline-flex items-center gap-1.5 px-2 py-1 min-w-0 max-w-full bg-ctp-blue/10 text-ctp-blue border border-ctp-blue/20"
              >
                <div class="flex-shrink-0">
                  <Activity size={12} />
                </div>
                <span class="text-ctp-text truncate min-w-0" title={item.name}>
                  {item.name}
                </span>
              </div>
            </div>

            <div
              class="flex items-center gap-1 text-xs text-ctp-lavender flex-shrink-0 sm:ml-2"
            >
              <span>{formatDate(item.createdAt)}</span>
              <span class="hidden sm:inline text-ctp-lavender/80">
                {formatTime(item.createdAt)}
              </span>
            </div>
          </div>
        </div>
      {/if}
    {/each}
  {/if}
</div>
