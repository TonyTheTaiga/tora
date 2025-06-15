<script lang="ts">
  import type { Experiment } from "$lib/types";
  import {
    getMode,
    addExperiment,
    selectedForComparison,
  } from "$lib/state/comparison.svelte.js";
  import {
    setSelectedExperiment,
    openEditExperimentModal,
    openDeleteExperimentModal,
  } from "$lib/state/app.svelte.js";
  import {
    Eye,
    EyeClosed,
    Globe,
    GlobeLock,
    Tag,
    Trash2,
    Edit,
  } from "lucide-svelte";
  import { page } from "$app/state";
  import { goto } from "$app/navigation";

  interface Props {
    experiments: Experiment[];
    highlighted: string[];
    onToggleHighlight: (experiment: Experiment) => void;
    formatDate: (date: Date) => string;
  }

  let { experiments, highlighted, onToggleHighlight, formatDate }: Props =
    $props();

  let currentWorkspace = $derived(page.data.currentWorkspace);
  let canDeleteExperiment = $derived(
    currentWorkspace && ["OWNER", "ADMIN"].includes(currentWorkspace.role),
  );
</script>

<div class="space-y-2 md:hidden font-mono">
  {#each experiments as experiment}
    <div
      class="group transition-colors
        {highlighted.length > 0 && !highlighted.includes(experiment.id)
        ? 'opacity-40'
        : ''}
        {selectedForComparison(experiment.id)
        ? 'bg-ctp-blue/10 border-l-2 border-ctp-blue'
        : ''}"
    >
      <div
        class="bg-ctp-surface0/10 backdrop-blur-md border border-ctp-surface0/20 p-3 hover:bg-ctp-surface0/20 transition-colors"
      >
        <button
          onclick={() => {
            if (getMode()) {
              addExperiment(experiment.id);
            } else {
              goto(
                `/workspaces/${page.params.slug}/experiments/${experiment.id}`,
              );
            }
          }}
          class="w-full text-left"
        >
          <!-- Header row -->
          <div class="flex items-center justify-between mb-2">
            <div class="flex items-center gap-2 min-w-0 flex-1">
              <div class="text-ctp-green text-xs"></div>
              <h3
                class="text-ctp-text group-hover:text-ctp-blue transition-colors truncate text-sm"
              >
                {experiment.name}
              </h3>
            </div>
            <div class="flex items-center gap-2 flex-shrink-0">
              {#if experiment.visibility === "PUBLIC"}
                <Globe class="w-3 h-3 text-ctp-green" />
              {:else}
                <GlobeLock class="w-3 h-3 text-ctp-red" />
              {/if}
              <span class="text-xs text-ctp-subtext0"
                >{formatDate(experiment.createdAt)}</span
              >
            </div>
          </div>

          <!-- Description -->
          {#if experiment.description}
            <p class="text-ctp-subtext1 text-xs mb-2 line-clamp-2">
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
              onToggleHighlight(experiment);
            }}
            class="bg-ctp-surface0/20 backdrop-blur-md border border-ctp-surface0/30 text-ctp-subtext0 hover:text-ctp-teal hover:border-ctp-teal/30 rounded-full p-2 text-xs transition-all"
            title="show experiment chain"
          >
            {#if highlighted.includes(experiment.id)}
              <EyeClosed class="w-4 h-4" />
            {:else}
              <Eye class="w-4 h-4" />
            {/if}
          </button>

          <button
            onclick={(e) => {
              e.stopPropagation();
              openEditExperimentModal(experiment);
            }}
            class="bg-ctp-surface0/20 backdrop-blur-md border border-ctp-surface0/30 text-ctp-subtext0 hover:text-ctp-blue hover:border-ctp-blue/30 rounded-full p-2 text-xs transition-all"
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
              class="bg-ctp-surface0/20 backdrop-blur-md border border-ctp-surface0/30 text-ctp-subtext0 hover:text-ctp-red hover:border-ctp-red/30 rounded-full p-2 text-xs transition-all"
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

