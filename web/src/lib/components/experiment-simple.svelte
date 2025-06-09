<script lang="ts">
  import type { Experiment } from "$lib/types";
  import {
    X,
    Tag,
    Clock,
    Eye,
    EyeClosed,
    Globe,
    GlobeLock,
    Settings,
  } from "lucide-svelte";
  import { page } from "$app/state";
  import { 
    openDeleteExperimentModal,
    setSelectedExperiment,
    getSelectedExperiment,
  } from "$lib/state/app.svelte.js";

  let {
    experiment,
    highlighted = $bindable(),
  }: {
    experiment: Experiment;
    highlighted: string[];
  } = $props();
</script>

<article
  class="flex flex-col h-full rounded-lg group hover:bg-ctp-surface0/30 transition-colors"
>
  <!-- Header -->
  <div class="flex items-center justify-between mb-2">
    <h3
      class="font-medium text-sm text-ctp-text group-hover:text-ctp-blue transition-colors truncate flex-1"
    >
      {experiment.name}
    </h3>
    <div class="flex-shrink-0">
      <div
        class="p-1 rounded-md transition-colors {experiment.visibility ===
        'PUBLIC'
          ? 'text-ctp-green hover:bg-ctp-green/10'
          : 'text-ctp-red hover:bg-ctp-red/10'}"
        title={experiment.visibility === "PUBLIC" ? "Public" : "Private"}
      >
        {#if experiment.visibility === "PUBLIC"}
          <Globe size={14} />
        {:else}
          <GlobeLock size={14} />
        {/if}
      </div>
    </div>
  </div>

  <!-- Description -->
  {#if experiment.description}
    <p
      class="text-xs text-ctp-subtext1 truncate mb-2"
      title={experiment.description}
    >
      {experiment.description}
    </p>
  {/if}

  <!-- Footer -->
  <div
    class="mt-auto flex items-center justify-between gap-2 pt-2 border-t border-ctp-surface1"
  >
    <!-- Tags -->
    <div
      class="flex items-center gap-1 text-xs text-ctp-subtext0 overflow-hidden"
    >
      {#if experiment.tags && experiment.tags.length > 0}
        <Tag size={10} class="text-ctp-overlay0 flex-shrink-0" />
        <div class="flex flex-nowrap gap-0.5 overflow-hidden">
          {#each experiment.tags as tag, i}
            <span class="text-[9px] text-ctp-overlay0 whitespace-nowrap">
              {tag}{i < experiment.tags.length - 1 ? ", " : ""}
            </span>
          {/each}
        </div>
      {/if}
    </div>

    <!-- Actions & Date -->
    <div class="flex items-center gap-1 text-ctp-subtext0">
      <button
        onclick={async (e) => {
          e.stopPropagation();
          if (highlighted.includes(experiment.id)) {
            highlighted = [];
          } else {
            try {
              const response = await fetch(
                `/api/experiments/${experiment.id}/ref`,
              );
              if (!response.ok) return;
              const data = await response.json();
              highlighted = [...data, experiment.id];
            } catch (err) {}
          }
        }}
        class="p-1 rounded-md hover:text-ctp-text hover:bg-ctp-surface0 transition-colors"
        title="Show experiment chain"
      >
        {#if highlighted.includes(experiment.id)}
          <EyeClosed size={14} />
        {:else}
          <Eye size={14} />
        {/if}
      </button>
      {#if page.data.user && page.data.user.id === experiment.user_id}
        <button
          type="button"
          class="p-1 rounded-md hover:text-ctp-red hover:bg-ctp-red/10 transition-colors"
          aria-label="Delete"
          title="Delete experiment"
          onclick={(e) => {
            e.stopPropagation();
            openDeleteExperimentModal(experiment);
          }}
        >
          <X size={14} />
        </button>
      {/if}
      {#if experiment?.createdAt}
        <time
          class="text-[10px] text-ctp-overlay0 ml-0.5"
          title={new Date(experiment.createdAt).toLocaleString()}
        >
          {new Date(experiment.createdAt).toLocaleDateString("en-US", {
            month: "short",
            day: "numeric",
          })}
        </time>
      {/if}
    </div>
  </div>
</article>
