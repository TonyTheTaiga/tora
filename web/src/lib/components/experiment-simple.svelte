<script lang="ts">
  import type { Experiment } from "$lib/types";
  import { X, Tag, Eye, EyeClosed, Globe, GlobeLock } from "lucide-svelte";
  import { page } from "$app/state";
  import { openDeleteExperimentModal } from "$lib/state/app.svelte.js";

  let {
    experiment,
    highlighted = $bindable(),
    isSelectedForComparison = false,
  }: {
    experiment: Experiment;
    highlighted: string[];
    isSelectedForComparison?: boolean;
  } = $props();

  let currentWorkspace = $derived(page.data.currentWorkspace);
  let canDeleteExperiment = $derived(
    page.data.user &&
      (page.data.user.id === experiment.user_id ||
        (currentWorkspace &&
          ["OWNER", "ADMIN"].includes(currentWorkspace.role))),
  );
</script>

<article
  class="flex flex-col h-full w-full rounded-xl bg-ctp-mantle p-3 shadow-md group hover:bg-ctp-surface0 transition-all duration-200 ease-in-out transform hover:-translate-y-1 border-2 md:h-52"
  class:border-ctp-blue={isSelectedForComparison}
  class:border-transparent={!isSelectedForComparison}
>
  <div class="flex items-start justify-between mb-2" data-testid="card-header">
    <div class="min-w-0 flex-1 pr-2">
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
        class="text-[11px] text-ctp-overlay1"
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

  {#if experiment.description}
    <p
      class="text-sm text-ctp-subtext0 mb-2 leading-relaxed flex-1 overflow-hidden description-truncate"
      title={experiment.description}
    >
      {experiment.description}
    </p>
  {/if}

  <div
    class="mt-auto flex items-center justify-between gap-1.5 pt-2 border-t border-ctp-surface1"
  >
    <div
      class="flex items-center gap-1 text-xs text-ctp-subtext0 overflow-x-auto md:overflow-visible min-w-0 pr-1.5 md:pr-0"
    >
      {#if experiment.tags && experiment.tags.length > 0}
        <Tag size={10} class="text-ctp-overlay1" />
        <div class="flex flex-nowrap md:flex-wrap gap-0.5 md:gap-1">
          {#each experiment.tags as tag}
            <span
              class="text-[10px] bg-ctp-blue/20 text-ctp-blue border border-ctp-blue/30 px-1 py-px rounded-full whitespace-nowrap inline-block max-w-[100px] truncate"
              title={tag}
            >
              {tag}
            </span>
          {/each}
        </div>
      {/if}
    </div>

    <div
      class="flex items-center gap-1 bg-ctp-surface0/40 backdrop-blur-sm border border-ctp-surface1/30 rounded-full p-0.5"
      data-testid="footer-actions-group"
    >
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
        class="p-1 rounded-full text-ctp-subtext0 hover:text-ctp-teal hover:bg-ctp-surface1/60 transition-colors"
        title="Show experiment chain"
      >
        {#if highlighted.includes(experiment.id)}
          <EyeClosed size={14} />
        {:else}
          <Eye size={14} />
        {/if}
      </button>
      <div
        class="p-1 rounded-full text-ctp-subtext0 transition-colors cursor-default"
        title={experiment.visibility === "PUBLIC" ? "Public" : "Private"}
        data-testid="visibility-status"
      >
        {#if experiment.visibility === "PUBLIC"}
          <Globe size={14} class="text-ctp-green" />
        {:else}
          <GlobeLock size={14} class="text-ctp-red" />
        {/if}
      </div>
      {#if canDeleteExperiment}
        <button
          type="button"
          class="p-1 rounded-full text-ctp-subtext0 hover:text-ctp-red hover:bg-ctp-surface1/60 transition-colors"
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
    </div>
  </div>
</article>

<style>
  .description-truncate {
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    overflow: hidden;
    text-overflow: ellipsis;
    min-height: 0;
  }

  @supports not (-webkit-line-clamp: 2) {
    .description-truncate {
      max-height: calc(1.5em * 2);
    }
  }

  .overflow-x-auto {
    scrollbar-width: thin;
    scrollbar-color: var(--color-ctp-surface2) var(--color-ctp-mantle);
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
