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
  import TerminalCard from "./TerminalCard.svelte";
  import EmptyState from "./EmptyState.svelte";

  interface Props {
    experiments: Experiment[];
    searchQuery?: string;
    showSummary?: boolean;
    formatDate?: (date: Date) => string;
  }

  let {
    experiments,
    searchQuery = "",
    showSummary = true,
    formatDate = (date: Date) =>
      date.toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year:
          date.getFullYear() !== new Date().getFullYear()
            ? "numeric"
            : undefined,
      }),
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

  function handleExperimentClick(experiment: Experiment) {
    if (getMode()) {
      addExperiment(experiment.id);
    } else {
      goto(`/experiments/${experiment.id}`);
    }
  }

  let selectedExperimentIds = $derived(
    filteredExperiments
      .filter((exp) => selectedForComparison(exp.id))
      .map((exp) => exp.id),
  );
</script>

{#if filteredExperiments.length === 0 && searchQuery}
  <EmptyState type="search" {searchQuery} />
{:else}
  <TerminalCard
    items={filteredExperiments}
    selectedItems={selectedExperimentIds}
  >
    {#snippet children(experiment)}
      <button
        onclick={() => handleExperimentClick(experiment)}
        class="flex items-center justify-between min-w-0 flex-1 group text-left"
      >
        <span
          class="text-ctp-text group-hover:text-ctp-blue transition-colors font-medium truncate"
        >
          {experiment.name}
        </span>
        <div class="flex items-center gap-2 text-xs text-ctp-subtext0">
          {#if experiment.tags && experiment.tags.length > 0}
            <div class="flex items-center gap-1">
              {#each experiment.tags.slice(0, 2) as tag}
                <span
                  class="text-[10px] bg-ctp-blue/20 text-ctp-blue border border-ctp-blue/30 px-1 py-px"
                >
                  {tag}
                </span>
              {/each}
              {#if experiment.tags.length > 2}
                <span class="text-[10px] text-ctp-subtext0"
                  >+{experiment.tags.length - 2}</span
                >
              {/if}
            </div>
            <span>|</span>
          {/if}
          <span>
            {formatDate(experiment.createdAt)}
          </span>
        </div>
      </button>
    {/snippet}

    {#snippet actions(experiment)}
      <div class="flex items-center justify-end gap-1">
        <button
          onclick={(e) => {
            e.stopPropagation();
            openEditExperimentModal(experiment);
          }}
          class="bg-ctp-surface0/20 backdrop-blur-md border border-ctp-surface0/30 text-ctp-subtext0 hover:text-ctp-blue hover:border-ctp-blue/30 rounded-full p-1 text-sm transition-all"
          title="edit experiment"
        >
          <Edit class="w-3 h-3" />
        </button>

        {#if canDeleteExperiment}
          <button
            onclick={(e) => {
              e.stopPropagation();
              openDeleteExperimentModal(experiment);
            }}
            class="bg-ctp-surface0/20 backdrop-blur-md border border-ctp-surface0/30 text-ctp-subtext0 hover:text-ctp-red hover:border-ctp-red/30 rounded-full p-1 text-sm transition-all"
            title="delete experiment"
          >
            <Trash2 class="w-3 h-3" />
          </button>
        {/if}
      </div>
    {/snippet}
  </TerminalCard>

  <!-- Summary line -->
  {#if showSummary}
    <div
      class="flex items-center text-sm text-ctp-subtext0 pt-2 border-t border-ctp-surface0/20 mt-4"
    >
      <div class="flex-1">
        {filteredExperiments.length} experiment{filteredExperiments.length !== 1
          ? "s"
          : ""} total
      </div>
    </div>
  {/if}
{/if}
