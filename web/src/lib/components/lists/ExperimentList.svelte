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
          const q = searchQuery.toLowerCase();
          return entry.name.includes(q);
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

  function getExperimentItemClass(): string {
    return "group layer-slide-up floating-element cursor-pointer relative mb-3 border-l-2 hover:border-l-ctp-blue/30";
  }
</script>

{#if filteredExperiments.length === 0 && searchQuery}
  <EmptyState type="search" {searchQuery} />
{:else}
  <ListCard
    items={filteredExperiments}
    getItemClass={getExperimentItemClass}
    {onItemClick}
  >
    {#snippet children(experiment)}
      <!-- Content -->
      <div class="flex-1 min-w-0">
        <!-- Header: Name and date -->
        <div class="flex items-center justify-between gap-3 mb-2">
          <h3
            class="text-ctp-text group-hover:text-ctp-blue transition-colors font-medium truncate"
          >
            {experiment.name}
          </h3>
          <div
            class="flex items-center gap-2 text-xs text-ctp-lavender flex-shrink-0"
          >
            <span>{formatDate(experiment.createdAt)}</span>
            <span class="text-ctp-lavender/80"
              >{formatTime(experiment.createdAt)}</span
            >
          </div>
        </div>

        <!-- Description -->
        {#if experiment.description}
          <p class="text-ctp-subtext1 text-sm mb-2">
            {experiment.description}
          </p>
        {/if}

        <!-- Tags and metadata -->
        <div class="flex items-center gap-2 text-xs">
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
            setExperimentToEdit(experiment);
          }}
          class="flex items-center gap-1 text-xs text-ctp-subtext0 hover:text-ctp-blue transition-colors bg-ctp-surface0/20 backdrop-blur-md border border-ctp-surface0/30 hover:border-ctp-blue/30 p-1"
          title="edit experiment"
        >
          <Edit class="w-3 h-3" />
          <span>Edit</span>
        </button>

        {#if canDeleteExperiment}
          <button
            onclick={(e) => {
              e.stopPropagation();
              setExperimentToDelete(experiment);
            }}
            class="flex items-center gap-1 text-xs text-ctp-subtext0 hover:text-ctp-red transition-colors bg-ctp-surface0/20 backdrop-blur-md border border-ctp-surface0/30 hover:border-ctp-red/30 p-1"
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
