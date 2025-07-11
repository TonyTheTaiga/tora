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
  import { Tag, Trash2, Edit } from "@lucide/svelte";
  import { page } from "$app/state";
  import { goto } from "$app/navigation";

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
</script>

{#if filteredExperiments.length === 0 && searchQuery}
  <div class="text-ctp-subtext0 text-base">
    <div>search "{searchQuery}"</div>
    <div class="text-ctp-subtext1 ml-2">no results found</div>
  </div>
{:else}
  <!-- Desktop table layout -->
  <div class="hidden md:block">
    <table class="w-full table-fixed">
      <thead>
        <tr class="text-sm text-ctp-subtext0 border-b border-ctp-surface0/20">
          <th class="text-left py-2 w-4"></th>
          <th class="text-left py-2">name</th>
          <th class="text-right py-2 w-24">modified</th>
          <th class="text-right py-2 w-32">actions</th>
        </tr>
      </thead>
      <tbody>
        {#each filteredExperiments as experiment}
          <tr
            class="group text-base hover:bg-ctp-surface0/20 transition-colors
              {selectedForComparison(experiment.id)
              ? 'bg-ctp-blue/10 border-l-2 border-ctp-blue'
              : ''}"
          >
            <td class="py-2 px-1">
              <div class="text-ctp-green text-sm"></div>
            </td>
            <td class="py-2 px-1 min-w-0">
              <button
                onclick={() => handleExperimentClick(experiment)}
                class="w-full text-left min-w-0"
              >
                <div class="truncate text-ctp-text">
                  <span
                    class="group-hover:text-ctp-blue transition-colors font-medium"
                  >
                    {experiment.name}
                  </span>
                  {#if experiment.description}
                    <span class="text-ctp-subtext1 text-sm">
                      - {experiment.description}
                    </span>
                  {/if}
                </div>
                {#if experiment.tags && experiment.tags.length > 0}
                  <div class="flex items-center gap-1 mt-1">
                    <Tag class="w-3 h-3 text-ctp-overlay1" />
                    <div class="flex gap-1">
                      {#each experiment.tags.slice(0, 3) as tag}
                        <span
                          class="text-[10px] bg-ctp-blue/20 text-ctp-blue border border-ctp-blue/30 px-1 py-px"
                        >
                          {tag}
                        </span>
                      {/each}
                      {#if experiment.tags.length > 3}
                        <span class="text-[10px] text-ctp-subtext0"
                          >+{experiment.tags.length - 3}</span
                        >
                      {/if}
                    </div>
                  </div>
                {/if}
              </button>
            </td>
            <td class="py-2 px-1 text-right text-sm text-ctp-subtext0 w-24">
              {formatDate(experiment.createdAt)}
            </td>
            <td class="py-2 px-1 text-right w-32">
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
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>

  <!-- Mobile card layout -->
  <div class="space-y-2 md:hidden">
    {#each filteredExperiments as experiment}
      <div
        class="group transition-colors
          {selectedForComparison(experiment.id)
          ? 'bg-ctp-blue/10 border-l-2 border-ctp-blue'
          : ''}"
      >
        <div
          class="bg-ctp-surface0/10 backdrop-blur-md border border-ctp-surface0/20 p-3 hover:bg-ctp-surface0/20 transition-colors"
        >
          <button
            onclick={() => handleExperimentClick(experiment)}
            class="w-full text-left"
          >
            <!-- Header row -->
            <div class="flex items-center justify-between mb-2">
              <div class="flex items-center gap-2 min-w-0 flex-1">
                <div class="text-ctp-green text-sm"></div>
                <h3
                  class="text-ctp-text group-hover:text-ctp-blue transition-colors truncate text-base"
                >
                  {experiment.name}
                </h3>
              </div>
              <div class="flex items-center gap-2 flex-shrink-0">
                <span class="text-sm text-ctp-subtext0"
                  >{formatDate(experiment.createdAt)}</span
                >
              </div>
            </div>

            <!-- Description -->
            {#if experiment.description}
              <p class="text-ctp-subtext1 text-sm mb-2 line-clamp-2">
                {experiment.description}
              </p>
            {/if}

            <!-- Tags -->
            {#if experiment.tags && experiment.tags.length > 0}
              <div class="flex items-center gap-1 mb-2">
                <Tag class="w-3 h-3 text-ctp-overlay1" />
                <div class="flex gap-1 flex-wrap">
                  {#each experiment.tags.slice(0, 3) as tag}
                    <span
                      class="text-[10px] bg-ctp-blue/20 text-ctp-blue border border-ctp-blue/30 px-1 py-px"
                    >
                      {tag}
                    </span>
                  {/each}
                  {#if experiment.tags.length > 3}
                    <span class="text-[10px] text-ctp-subtext0"
                      >+{experiment.tags.length - 3}</span
                    >
                  {/if}
                </div>
              </div>
            {/if}
          </button>

          <!-- Actions row -->
          <div
            class="flex items-center justify-end gap-1 pt-2 border-t border-ctp-surface0/20"
          >
            <button
              onclick={(e) => {
                e.stopPropagation();
                openEditExperimentModal(experiment);
              }}
              class="bg-ctp-surface0/20 backdrop-blur-md border border-ctp-surface0/30 text-ctp-subtext0 hover:text-ctp-blue hover:border-ctp-blue/30 rounded-full p-2 text-sm transition-all"
              title="edit experiment"
            >
              <Edit class="w-4 h-4" />
            </button>

            {#if canDeleteExperiment}
              <button
                onclick={(e) => {
                  e.stopPropagation();
                  openDeleteExperimentModal(experiment);
                }}
                class="bg-ctp-surface0/20 backdrop-blur-md border border-ctp-surface0/30 text-ctp-subtext0 hover:text-ctp-red hover:border-ctp-red/30 rounded-full p-2 text-sm transition-all"
                title="delete experiment"
              >
                <Trash2 class="w-4 h-4" />
              </button>
            {/if}
          </div>
        </div>
      </div>
    {/each}
  </div>

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
