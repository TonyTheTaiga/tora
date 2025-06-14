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

<div class="hidden md:block space-y-1 font-mono">
  <!-- Header -->
  <div
    class="flex items-center text-xs text-ctp-subtext0 pb-2 border-b border-ctp-surface0/20"
  >
    <div class="w-4">•</div>
    <div class="flex-1">name</div>
    <div class="w-16 text-right">visibility</div>
    <div class="w-20 text-right">modified</div>
    <div class="w-20 text-right">actions</div>
  </div>

  <!-- Experiment entries -->
  {#each experiments as experiment}
    <div
      class="group flex items-center text-sm hover:bg-ctp-surface0/20 px-1 py-2 transition-colors
        {highlighted.length > 0 && !highlighted.includes(experiment.id)
        ? 'opacity-40'
        : ''}
        {selectedForComparison(experiment.id)
        ? 'bg-ctp-blue/10 border-l-2 border-ctp-blue'
        : ''}"
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
        class="flex items-center flex-1 min-w-0 text-left"
      >
        <div class="w-4 text-ctp-green text-xs">●</div>
        <div class="flex-1 min-w-0">
          <div class="flex items-center gap-2">
            <span
              class="text-ctp-text group-hover:text-ctp-blue transition-colors truncate"
            >
              {experiment.name}
            </span>
            {#if experiment.description}
              <span class="text-ctp-subtext1 text-xs truncate">
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
        </div>
      </button>

      <div class="w-16 text-right text-xs">
        {#if experiment.visibility === "PUBLIC"}
          <Globe class="w-3 h-3 text-ctp-green inline" />
        {:else}
          <GlobeLock class="w-3 h-3 text-ctp-red inline" />
        {/if}
      </div>

      <div class="w-20 text-right text-xs text-ctp-subtext0">
        {formatDate(experiment.createdAt)}
      </div>

      <div class="w-20 text-right">
        <div class="flex items-center justify-end gap-1">
          <button
            onclick={(e) => {
              e.stopPropagation();
              onToggleHighlight(experiment);
            }}
            class="bg-ctp-surface0/20 backdrop-blur-md border border-ctp-surface0/30 text-ctp-subtext0 hover:text-ctp-teal hover:border-ctp-teal/30 rounded-full p-1 text-xs transition-all"
            title="show experiment chain"
          >
            {#if highlighted.includes(experiment.id)}
              <EyeClosed class="w-3 h-3" />
            {:else}
              <Eye class="w-3 h-3" />
            {/if}
          </button>

          <button
            onclick={(e) => {
              e.stopPropagation();
              openEditExperimentModal(experiment);
            }}
            class="bg-ctp-surface0/20 backdrop-blur-md border border-ctp-surface0/30 text-ctp-subtext0 hover:text-ctp-blue hover:border-ctp-blue/30 rounded-full p-1 text-xs transition-all"
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
              class="bg-ctp-surface0/20 backdrop-blur-md border border-ctp-surface0/30 text-ctp-subtext0 hover:text-ctp-red hover:border-ctp-red/30 rounded-full p-1 text-xs transition-all"
              title="delete experiment"
            >
              <Trash2 class="w-3 h-3" />
            </button>
          {/if}
        </div>
      </div>
    </div>
  {/each}
</div>
