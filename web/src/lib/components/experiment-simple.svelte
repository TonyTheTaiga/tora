<script lang="ts">
  import type { Experiment } from "$lib/types";
  import {
    X,
    Tag,
    Clock,
    ChartLine,
    Eye,
    EyeClosed,
    Globe,
    GlobeLock,
  } from "lucide-svelte";
  import { page } from "$app/state";

  let {
    experiment,
    selectedExperiment = $bindable(),
    highlighted = $bindable(),
    selectedForDelete = $bindable(),
  }: {
    experiment: Experiment;
    selectedExperiment: Experiment | null;
    highlighted: string[];
    selectedForDelete: Experiment | null;
  } = $props();
</script>

<article class="flex flex-col h-full group">
  <!-- Header -->
  <div class="flex justify-between items-center mb-3">
    <h3
      class="font-semibold text-sm text-ctp-text truncate pr-2 group-hover:text-ctp-blue transition-colors"
    >
      {experiment.name}
    </h3>
    <div class="flex items-center gap-0.5">
      <div
        class="p-1 rounded-md transition-colors {experiment.visibility ===
        'PUBLIC'
          ? 'text-ctp-green hover:bg-ctp-green/10'
          : 'text-ctp-red hover:bg-ctp-red/10'}"
      >
        {#if experiment.visibility === "PUBLIC"}
          <Globe size={14} />
        {:else}
          <GlobeLock size={14} />
        {/if}
      </div>
      <button
        onclick={async () => {
          if (highlighted.includes(experiment.id)) {
            highlighted = [];
          } else {
            try {
              const response = await fetch(
                `/api/experiments/${experiment.id}/ref`,
              );
              if (!response.ok) {
                return;
              }
              const data = await response.json();
              highlighted = [...data, experiment.id];
            } catch (err) {}
          }
        }}
        class="p-1 rounded-md text-ctp-subtext0 hover:text-ctp-text hover:bg-ctp-surface0 transition-all"
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
          class="p-1 rounded-md text-ctp-subtext0 hover:text-ctp-red hover:bg-ctp-red/10 transition-all"
          aria-label="Delete"
          title="Delete experiment"
          onclick={(e) => {
            e.stopPropagation();
            selectedForDelete = experiment;
          }}
        >
          <X size={14} />
        </button>
      {/if}
    </div>
  </div>

  <!-- Main -->
  <div class="flex-grow flex flex-col">
    <!-- Description -->
    {#if experiment.description}
      <p class="text-ctp-subtext0 text-xs leading-relaxed mb-3 line-clamp-2">
        {experiment.description}
      </p>
    {:else}
      <p class="text-ctp-overlay0 text-xs italic mb-3">No description</p>
    {/if}

    <!-- Metrics indicator -->
    <div class="flex items-center gap-1.5 text-ctp-subtext1 text-xs">
      <ChartLine size={12} />
      <span class="font-medium">
        {experiment.availableMetrics.length} metric{experiment.availableMetrics
          .length !== 1
          ? "s"
          : ""}
      </span>
    </div>
  </div>

  <!-- Footer -->
  <div
    class="mt-auto flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-ctp-surface0/50"
  >
    <!-- Tags -->
    {#if experiment.tags && experiment.tags.length > 0}
      <div class="flex items-center gap-1 text-xs text-ctp-subtext0">
        <Tag size={10} class="text-ctp-overlay0" />
        {#each experiment.tags.slice(0, 2) as tag, i}
          <span
            class="px-1.5 py-0.5 bg-ctp-lavender/10 text-ctp-lavender rounded-full text-[10px] font-medium"
          >
            {tag}
          </span>
          {#if i === 0 && experiment.tags.length > 2}
            <span class="text-ctp-overlay0 text-[10px]"
              >+{experiment.tags.length - 1}</span
            >
          {/if}
        {/each}
      </div>
    {:else}
      <div></div>
    {/if}

    <!-- Created At -->
    {#if experiment?.createdAt}
      <time class="flex items-center gap-1 text-[10px] text-ctp-overlay0">
        <Clock size={10} />
        {new Date(experiment.createdAt).toLocaleDateString("en-US", {
          month: "short",
          day: "numeric",
        })}
      </time>
    {/if}
  </div>
</article>
