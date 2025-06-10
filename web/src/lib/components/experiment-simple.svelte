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
    isSelectedForComparison = false, // New prop
  }: {
    experiment: Experiment;
    highlighted: string[];
    isSelectedForComparison?: boolean; // Optional prop
  } = $props();
</script>

<article
  class="flex flex-col h-full rounded-xl bg-ctp-mantle p-4 shadow-md group hover:bg-ctp-surface0 transition-all duration-200 ease-in-out transform hover:-translate-y-1 border-2"
  class:border-ctp-blue={isSelectedForComparison}
  class:border-transparent={!isSelectedForComparison}
>
  <!-- Header -->
  <div class="flex items-start justify-between mb-3">
    <h3
      class="font-semibold text-base text-ctp-text group-hover:text-ctp-blue transition-colors flex-1 mr-2"
    >
      {experiment.name}
    </h3>
    <div class="flex-shrink-0">
      <div
        class="p-1 rounded-md transition-colors {experiment.visibility ===
        'PUBLIC'
          ? 'text-ctp-green bg-ctp-green/10 hover:bg-ctp-green/20'
          : 'text-ctp-red bg-ctp-red/10 hover:bg-ctp-red/20'}"
        title={experiment.visibility === "PUBLIC" ? "Public" : "Private"}
      >
        {#if experiment.visibility === "PUBLIC"}
          <Globe size={16} />
        {:else}
          <GlobeLock size={16} />
        {/if}
      </div>
    </div>
  </div>

  <!-- Description -->
  {#if experiment.description}
    <p
      class="text-sm text-ctp-subtext0 mb-3 leading-relaxed"
      title={experiment.description}
    >
      {experiment.description}
    </p>
  {/if}

  <!-- Footer -->
  <div
    class="mt-auto flex items-center justify-between gap-2 pt-3 border-t border-ctp-surface1"
  >
    <!-- Tags -->
    <div
      class="flex items-center gap-1.5 text-xs text-ctp-subtext0 overflow-hidden"
    >
      {#if experiment.tags && experiment.tags.length > 0}
        <Tag size={12} class="text-ctp-overlay1 flex-shrink-0" />
        <div class="flex flex-wrap gap-1 overflow-hidden">
          {#each experiment.tags as tag, i}
            <span
              class="text-xs bg-ctp-surface0 text-ctp-overlay2 px-1.5 py-0.5 rounded-full whitespace-nowrap"
            >
              {tag}
            </span>
          {/each}
        </div>
      {/if}
    </div>

    <!-- Actions & Date -->
    <div class="flex items-center gap-2 text-ctp-subtext0">
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
        class="p-1.5 rounded-lg hover:text-ctp-text hover:bg-ctp-surface1 transition-colors"
        title="Show experiment chain"
      >
        {#if highlighted.includes(experiment.id)}
          <EyeClosed size={16} />
        {:else}
          <Eye size={16} />
        {/if}
      </button>
      {#if page.data.user && page.data.user.id === experiment.user_id}
        <button
          type="button"
          class="p-1.5 rounded-lg hover:text-ctp-red hover:bg-ctp-red/20 transition-colors"
          aria-label="Delete"
          title="Delete experiment"
          onclick={(e) => {
            e.stopPropagation();
            openDeleteExperimentModal(experiment);
          }}
        >
          <X size={16} />
        </button>
      {/if}
      {#if experiment?.createdAt}
        <time
          class="text-xs text-ctp-overlay1 ml-1"
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
