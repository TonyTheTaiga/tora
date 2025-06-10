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
  class="flex flex-col h-full w-full rounded-xl bg-ctp-mantle p-3 shadow-md group hover:bg-ctp-surface0 transition-all duration-200 ease-in-out transform hover:-translate-y-1 border-2 md:h-52"
  class:border-ctp-blue={isSelectedForComparison}
  class:border-transparent={!isSelectedForComparison}
>
  <!-- Header -->
  <div class="flex items-start justify-between mb-2 flex-shrink-0" data-testid="card-header">
    <div class="min-w-0 flex-grow pr-2">
      <h3
        class="font-semibold text-sm text-ctp-text group-hover:text-ctp-blue transition-colors truncate"
        title={experiment.name}
        data-testid="experiment-name"
      >
        {experiment.name}
      </h3>
    </div>
    {#if experiment?.createdAt}
      <time
        class="text-[11px] text-ctp-overlay1 flex-shrink-0"
        title={new Date(experiment.createdAt).toLocaleString()}
        data-testid="experiment-date"
      >
        {new Date(experiment.createdAt).toLocaleDateString("en-US", {
          month: "short",
          day: "numeric",
        })}
      </time>
    {/if}
  </div>

  <!-- Description -->
  {#if experiment.description}
    <p
      class="text-sm text-ctp-subtext0 mb-2 leading-relaxed flex-grow overflow-hidden description-truncate"
      title={experiment.description}
    >
      {experiment.description}
    </p>
  {/if}

  <!-- Footer -->
  <div
    class="mt-auto flex items-center justify-between gap-1.5 pt-2 border-t border-ctp-surface1 flex-shrink-0"
  >
    <!-- Tags -->
    <div
      class="flex items-center gap-1 text-xs text-ctp-subtext0 overflow-x-auto md:overflow-visible min-w-0 pr-1.5 md:pr-0" /* Adjusted overflow and padding for md screens */
    >
      {#if experiment.tags && experiment.tags.length > 0}
        <Tag size={10} class="text-ctp-overlay1 flex-shrink-0" />
        <div class="flex flex-nowrap md:flex-wrap gap-0.5 md:gap-1"> {/* Adjusted flex behavior and gap for md screens */}
          {#each experiment.tags as tag, i}
            <span
              class="text-[10px] bg-ctp-surface0 text-ctp-overlay2 px-1 py-px rounded-full whitespace-nowrap inline-block max-w-[100px] truncate"
              title={tag}
            >
              {tag}
            </span>
          {/each}
        </div>
      {/if}
    </div>

    <!-- Actions, Date & Visibility -->
    <div class="flex items-center gap-1.5 text-ctp-subtext0 flex-shrink-0" data-testid="footer-actions-group">
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
        class="p-1 rounded-md hover:text-ctp-text hover:bg-ctp-surface1 transition-colors"
        title="Show experiment chain"
      >
        {#if highlighted.includes(experiment.id)}
          <EyeClosed size={14} />
        {:else}
          <Eye size={14} />
        {/if}
      </button>
      <!-- Visibility Icon moved here -->
      <div
        class="p-1 rounded-md hover:bg-ctp-surface1 transition-colors cursor-default" /* Adjusted padding and added hover, cursor-default if not clickable */
        title={experiment.visibility === "PUBLIC" ? "Public" : "Private"}
        data-testid="visibility-status"
      >
        {#if experiment.visibility === "PUBLIC"}
          <Globe size={14} class="text-ctp-green" />
        {:else}
          <GlobeLock size={14} class="text-ctp-red" />
        {/if}
      </div>
      {#if page.data.user && page.data.user.id === experiment.user_id}
        <button
          type="button"
          class="p-1 rounded-md hover:text-ctp-red hover:bg-ctp-red/20 transition-colors"
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
      <!-- Date element removed from here -->
    </div>
  </div>
</article>

<style>
  .description-truncate {
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: 2; /* Reduced to 2 lines */
    overflow: hidden;
    text-overflow: ellipsis;
    min-height: 0; /* Important for flex-grow in some browsers */
  }

  /* Fallback for non-webkit browsers for description, not perfect but better than nothing */
  @supports not (-webkit-line-clamp: 2) {
    .description-truncate {
      max-height: calc(1.5em * 2); /* Assuming line-height is around 1.5em, for 2 lines */
    }
  }

  /* Custom scrollbar for tags (already present, ensure it's still relevant) */
  .overflow-x-auto {
    scrollbar-width: thin; /* For Firefox */
    scrollbar-color: var(--color-ctp-surface2) var(--color-ctp-mantle); /* For Firefox */
  }
  .overflow-x-auto::-webkit-scrollbar {
    height: 4px;
  }
  .overflow-x-auto::-webkit-scrollbar-track {
    background: var(--color-ctp-mantle);
    border-radius: 2px;
  }
  .overflow-x-auto::-webkit-scrollbar-thumb {
    background-color: var(--color-ctp-surface2);
    border-radius: 2px;
  }
</style>
