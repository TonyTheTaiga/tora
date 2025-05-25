<script lang="ts">
  import type { Experiment, ExperimentStatus } from "$lib/types"; // Import ExperimentStatus
  import {
    X,
    Tag,
    Clock,
    // ChartLine, // Removed: No longer showing number of metrics
    Eye,
    EyeClosed,
    Globe,
    GlobeLock,
  } from "lucide-svelte";
  import { page } from "$app/state";

  let {
    experiment,
    // selectedExperiment = $bindable(), // This prop is not used in the provided simple template, assuming it's for click handling handled by parent
    highlighted = $bindable(),
    selectedForDelete = $bindable(),
  }: {
    experiment: Experiment; // Assumes Experiment type now includes status and keyMetrics
    // selectedExperiment: Experiment | null;
    highlighted: string[];
    selectedForDelete: Experiment | null;
  } = $props();

  // Define a mapping for status colors (Tailwind CSS classes)
  const statusColors: Record<ExperimentStatus, string> = {
    COMPLETED: "bg-ctp-green",
    RUNNING: "bg-ctp-blue",
    FAILED: "bg-ctp-red",
    DRAFT: "bg-ctp-overlay1",
    OTHER: "bg-ctp-subtext0",
  };

  const statusTooltips: Record<ExperimentStatus, string> = {
    COMPLETED: "Completed",
    RUNNING: "Running",
    FAILED: "Failed",
    DRAFT: "Draft",
    OTHER: "Other",
  };

  let currentStatusColor = $derived(
    experiment.status ? statusColors[experiment.status] : "bg-ctp-subtext0",
  );
  let currentStatusTooltip = $derived(
    experiment.status ? statusTooltips[experiment.status] : "Unknown Status",
  );
</script>

<article class="flex flex-col h-full group">
  <!-- TOP: Name, Status, Visibility -->
  <div class="flex justify-between items-start mb-2">
    <h3
      class="font-semibold text-sm text-ctp-text group-hover:text-ctp-blue transition-colors line-clamp-2 pr-2 break-all"
    >
      {experiment.name}
    </h3>
    <div class="flex items-center gap-1.5 flex-shrink-0">
      <!-- Status Indicator -->
      {#if experiment.status}
        <span
          class="w-3 h-3 rounded-full {currentStatusColor}"
          title={currentStatusTooltip}
        ></span>
      {/if}
      <!-- Visibility Icon -->
      <div
        class="p-0.5 rounded-md transition-colors {experiment.visibility ===
        'PUBLIC'
          ? 'text-ctp-green hover:bg-ctp-green/10'
          : 'text-ctp-red hover:bg-ctp-red/10'}"
        title={experiment.visibility === 'PUBLIC' ? 'Public' : 'Private'}
      >
        {#if experiment.visibility === "PUBLIC"}
          <Globe size={14} />
        {:else}
          <GlobeLock size={14} />
        {/if}
      </div>
    </div>
  </div>

  <!-- MIDDLE: Key Metrics -->
  <div class="mb-2 flex flex-col gap-1">
    {#if experiment.keyMetrics && experiment.keyMetrics.length > 0}
      {#each experiment.keyMetrics as metric (metric.name)}
        <div class="text-xs flex justify-between items-center">
          <span class="text-ctp-subtext1 truncate pr-1">{metric.name}:</span>
          <span class="font-medium text-ctp-text ml-1 truncate">{metric.value}</span>
        </div>
      {/each}
    {:else}
      <p class="text-ctp-overlay0 text-xs italic">No key metrics.</p> <!-- Adjusted placeholder text -->
    {/if}
  </div>

  <!-- BOTTOM: Tags, Date, Actions -->
  <div
    class="mt-auto flex flex-wrap items-center justify-between gap-x-2 gap-y-1 pt-2 border-t border-ctp-surface0/50"
  >
    <!-- Tags -->
    <div class="flex items-center gap-1 text-xs text-ctp-subtext0 min-w-[60px]">
      {#if experiment.tags && experiment.tags.length > 0}
        <Tag size={10} class="text-ctp-overlay0 flex-shrink-0" />
        <div class="flex flex-wrap gap-0.5 items-center">
          {#each experiment.tags.slice(0, 2) as tag, i}
            <span
              class="px-1 py-0.5 bg-ctp-lavender/10 text-ctp-lavender rounded-full text-[9px] font-medium leading-none"
            >
              {tag}
            </span>
            {#if i === 0 && experiment.tags.length > 2}
              <span class="text-ctp-overlay0 text-[9px]"
                >+{experiment.tags.length - 1}</span
              >
            {/if}
          {/each}
        </div>
      {:else}
        <div class="h-[18px]"></div> <!-- Placeholder to maintain height consistency if no tags -->
      {/if}
    </div>
    <!-- Actions & Date -->
    <div class="flex items-center gap-1 text-ctp-subtext0 flex-shrink-0">
      <button
        on:click={async (e) => { // Changed to on:click
          e.stopPropagation(); // Prevent card click
          if (highlighted.includes(experiment.id)) {
            highlighted = [];
          } else {
            try {
              // Assuming this API endpoint exists and returns string[]
              const response = await fetch(
                `/api/experiments/${experiment.id}/ref`,
              );
              if (!response.ok) {
                return;
              }
              const data = await response.json();
              highlighted = [...data, experiment.id];
            } catch (err) {
              // console.error("Failed to fetch experiment chain:", err);
            }
          }
        }}
        class="p-1 rounded-md hover:text-ctp-text hover:bg-ctp-surface0 transition-all"
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
          class="p-1 rounded-md hover:text-ctp-red hover:bg-ctp-red/10 transition-all"
          aria-label="Delete"
          title="Delete experiment"
          on:click={(e) => { // Changed to on:click
            e.stopPropagation(); // Prevent card click
            selectedForDelete = experiment;
          }}
        >
          <X size={14} />
        </button>
      {/if}
      {#if experiment?.createdAt}
        <time class="flex items-center gap-1 text-[10px] text-ctp-overlay0 ml-0.5 whitespace-nowrap">
          <Clock size={10} />
          {new Date(experiment.createdAt).toLocaleDateString("en-US", {
            month: "short",
            day: "numeric",
          })}
        </time>
      {/if}
    </div>
  </div>
</article>
