<script lang="ts">
  import type { Experiment } from "$lib/types";
  import {
    setExperimentToEdit,
    setExperimentToDelete,
  } from "$lib/state/modal.svelte.js";
  import { Trash2, Edit } from "@lucide/svelte";
  import { page } from "$app/state";
  import EmptyState from "./EmptyState.svelte";
  import ListCard from "./ListCard.svelte";
  import ListActionsMenu, { type MenuItem } from "./ListActionsMenu.svelte";

  let { experiments, searchQuery, onItemClick } = $props();

  function formatDate(date: Date) {
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year:
        date.getFullYear() !== new Date().getFullYear() ? "numeric" : undefined,
    });
  }

  let currentWorkspace = $derived(page.data.currentWorkspace);
  let canDeleteExperiment = $derived(
    currentWorkspace && ["OWNER", "ADMIN"].includes(currentWorkspace.role),
  );

  let filteredExperiments = $derived(
    experiments
      .map((exp: Experiment) => ({
        exp,
        name: exp.name.toLowerCase(),
        desc: exp.description?.toLowerCase() ?? "",
        tags: exp.tags?.map((t: string) => t.toLowerCase()) ?? [],
      }))
      .filter(
        (entry: {
          exp: Experiment;
          name: string;
          desc: string;
          tags: string[];
        }) => {
          if (!searchQuery) return true;
          const terms = searchQuery.toLowerCase().split(/\s+/).filter(Boolean);
          return terms.every(
            (t: string) =>
              entry.name.includes(t) ||
              entry.desc.includes(t) ||
              entry.tags.some((tag: string) => tag.includes(t)),
          );
        },
      )
      .map((e: any) => e.exp),
  );

  function formatTime(date: Date): string {
    return date.toLocaleTimeString("en-US", {
      hour: "numeric",
      minute: "2-digit",
      hour12: true,
    });
  }
</script>

{#if filteredExperiments.length === 0 && searchQuery}
  <EmptyState type="search" {searchQuery} />
{:else}
  <ListCard items={filteredExperiments} {onItemClick}>
    {#snippet children(experiment)}
      <div class="flex items-center justify-between gap-3 mb-2">
        <h3
          class="text-ctp-text group-hover:text-ctp-blue transition-colors font-medium truncate"
        >
          {experiment.name}
        </h3>
        <div class="flex items-center gap-2 text-xs text-ctp-lavender">
          <span>{formatDate(experiment.createdAt)}</span>
          <span class="text-ctp-lavender/80"
            >{formatTime(experiment.createdAt)}</span
          >
        </div>
      </div>

      {#if experiment.description}
        <p class="text-ctp-subtext1 text-sm mb-2">
          {experiment.description}
        </p>
      {/if}

      <div class="flex items-center gap-2 text-xs">
        {#if experiment.tags && experiment.tags.length > 0}
          <div class="flex items-center gap-1.5 flex-wrap">
            {#each experiment.tags.slice(0, 3) as tag}
              <span
                class="bg-ctp-surface0/30 text-ctp-subtext0 border border-ctp-surface0/40 rounded-sm px-2 py-0.5 text-[10px]"
              >
                {tag}
              </span>
            {/each}
            {#if experiment.tags.length > 3}
              <span class="text-ctp-subtext1 text-[10px]">
                +{experiment.tags.length - 3}
              </span>
            {/if}
          </div>
        {/if}

        <div class="flex items-center gap-2 text-ctp-lavender">
          {#if experiment.hyperparams && experiment.hyperparams.length > 0}
            <span>
              {experiment.hyperparams.length} param{experiment.hyperparams
                .length !== 1
                ? "s"
                : ""}
            </span>
          {/if}

          {#if experiment.availableMetrics && experiment.availableMetrics.length > 0}
            {#if experiment.hyperparams && experiment.hyperparams.length > 0}
              <span>•</span>
            {/if}
            <span>
              {experiment.availableMetrics.length} metric{experiment
                .availableMetrics.length !== 1
                ? "s"
                : ""}
            </span>
          {/if}
        </div>
      </div>
    {/snippet}

    {#snippet actions(experiment)}
      <ListActionsMenu
        ariaLabel="experiment actions"
        items={[
          {
            label: "Edit",
            icon: Edit,
            onSelect: () => setExperimentToEdit(experiment),
          },
          ...(canDeleteExperiment
            ? [
                { type: "separator" } as const,
                {
                  label: "Delete",
                  icon: Trash2,
                  destructive: true,
                  onSelect: () => setExperimentToDelete(experiment),
                },
              ]
            : []),
        ] satisfies MenuItem[]}
      />
    {/snippet}
  </ListCard>
{/if}
