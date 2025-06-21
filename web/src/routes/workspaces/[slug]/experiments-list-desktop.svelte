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
  import { Tag, Trash2, Edit } from "lucide-svelte";
  import { page } from "$app/state";
  import { goto } from "$app/navigation";

  interface Props {
    experiments: Experiment[];
    formatDate: (date: Date) => string;
  }

  let { experiments, formatDate }: Props = $props();

  let currentWorkspace = $derived(page.data.currentWorkspace);
  let canDeleteExperiment = $derived(
    currentWorkspace && ["OWNER", "ADMIN"].includes(currentWorkspace.role),
  );
</script>

<div class="hidden md:block font-mono">
  <table class="w-full table-fixed">
    <thead>
      <tr class="text-sm text-ctp-subtext0 border-b border-ctp-surface0/20">
        <th class="text-left py-2 w-4">•</th>
        <th class="text-left py-2">name</th>
        <th class="text-right py-2 w-24">modified</th>
        <th class="text-right py-2 w-32">actions</th>
      </tr>
    </thead>
    <tbody>
      {#each experiments as experiment}
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
              onclick={() => {
                if (getMode()) {
                  addExperiment(experiment.id);
                } else {
                  goto(`/experiments/${experiment.id}`);
                }
              }}
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
