<script lang="ts">
  import type { Experiment } from "$lib/types";
  import {
    getMode,
    addExperiment,
    selectedForComparison,
  } from "$lib/state/comparison.svelte.js";
  import {
    openEditExperimentModal,
    openDeleteExperimentModal,
  } from "$lib/state/app.svelte.js";
  import { Trash2, Edit } from "@lucide/svelte";
  import { page } from "$app/state";
  import { goto } from "$app/navigation";
  import EmptyState from "./EmptyState.svelte";
  import ListCard from "./ListCard.svelte";

  interface Props {
    experiments: Experiment[];
    searchQuery?: string;
    formatDate?: (date: Date) => string;
    onItemClick?: (experiment: Experiment) => void;
  }

  let {
    experiments,
    searchQuery = "",
    formatDate = (date: Date) =>
      date.toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year:
          date.getFullYear() !== new Date().getFullYear()
            ? "numeric"
            : undefined,
      }),
    onItemClick = (experiment: Experiment) => {
      if (getMode()) {
        addExperiment(experiment.id);
      } else {
        goto(`/experiments/${experiment.id}`);
      }
    },
  }: Props = $props();

  let currentWorkspace = $derived(page.data.currentWorkspace);
  let canDeleteExperiment = $derived(
    currentWorkspace && ["OWNER", "ADMIN"].includes(currentWorkspace.role),
  );

  let filteredExperiments = $derived(
    experiments
      .map((exp) => ({
        exp,
        name: exp.name.toLowerCase(),
        desc: exp.description?.toLowerCase() ?? "",
        tags: exp.tags?.map((t: string) => t.toLowerCase()) ?? [],
      }))
      .filter((entry) => {
        if (!searchQuery) return true;
        const q = searchQuery.toLowerCase();
        return entry.name.includes(q);
      })
      .map((e) => e.exp),
  );

  let selectedExperimentIds = $derived(
    filteredExperiments
      .filter((exp) => selectedForComparison(exp.id))
      .map((exp) => exp.id),
  );

  function isSelected(experimentId: string): boolean {
    return selectedExperimentIds.includes(experimentId);
  }

  function formatTime(date: Date): string {
    return date.toLocaleTimeString("en-US", {
      hour: "numeric",
      minute: "2-digit",
      hour12: true,
    });
  }

  function getTimelineItemClass(experiment: Experiment): string {
    const baseClass = "group layer-slide-up";
    const selectedClass = isSelected(experiment.id)
      ? "surface-accent-blue"
      : "surface-interactive";

    return `${baseClass} ${selectedClass}`.trim();
  }
</script>

{#if filteredExperiments.length === 0 && searchQuery}
  <EmptyState type="search" {searchQuery} />
{:else}
  <ListCard
    items={filteredExperiments}
    getItemClass={getTimelineItemClass}
    {onItemClick}
  >
    {#snippet children(experiment)}
      <!-- Content - Mobile-first responsive layout -->
      <div class="flex-1 min-w-0">
        <!-- Header: Name and date -->
        <div
          class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1 sm:gap-3 mb-2"
        >
          <h3
            class="text-ctp-text group-hover:text-ctp-blue transition-colors font-medium truncate"
          >
            {experiment.name}
          </h3>
          <div
            class="flex items-center gap-2 text-xs text-ctp-lavender flex-shrink-0"
          >
            <span>{formatDate(experiment.createdAt)}</span>
            <span class="hidden sm:inline text-ctp-lavender/80"
              >{formatTime(experiment.createdAt)}</span
            >
          </div>
        </div>

        <!-- Description -->
        {#if experiment.description}
          <p
            class="text-ctp-subtext1 text-sm mb-2 line-clamp-2 sm:line-clamp-none"
          >
            {experiment.description}
          </p>
        {/if}

        <!-- Tags and metadata - Stack on mobile -->
        <div class="flex flex-col sm:flex-row sm:items-center gap-2 text-xs">
          <!-- Tags -->
          {#if experiment.tags && experiment.tags.length > 0}
            <div class="flex items-center gap-1 flex-wrap">
              {#each experiment.tags.slice(0, 3) as tag}
                <span
                  class="bg-ctp-surface0/30 text-ctp-subtext0 px-2 py-1 text-[10px]"
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

          <!-- Metadata -->
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
      </div>
    {/snippet}

    {#snippet actions(experiment)}
      <div class="flex items-center justify-end gap-2">
        <button
          onclick={(e) => {
            e.stopPropagation();
            openEditExperimentModal(experiment);
          }}
          class="flex items-center gap-1 text-xs text-ctp-subtext0 hover:text-ctp-blue transition-colors sm:bg-ctp-surface0/20 sm:backdrop-blur-md sm:border sm:border-ctp-surface0/30 sm:hover:border-ctp-blue/30 sm:p-1"
          title="edit experiment"
        >
          <Edit class="w-3 h-3" />
          <span>Edit</span>
        </button>

        {#if canDeleteExperiment}
          <button
            onclick={(e) => {
              e.stopPropagation();
              openDeleteExperimentModal(experiment);
            }}
            class="flex items-center gap-1 text-xs text-ctp-subtext0 hover:text-ctp-red transition-colors sm:bg-ctp-surface0/20 sm:backdrop-blur-md sm:border sm:border-ctp-surface0/30 sm:hover:border-ctp-red/30 sm:p-1"
            title="delete experiment"
          >
            <Trash2 class="w-3 h-3" />
            <span>Delete</span>
          </button>
        {/if}
      </div>
    {/snippet}
  </ListCard>
{/if}
