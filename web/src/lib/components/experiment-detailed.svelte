<script lang="ts">
  import type {
    Experiment,
    ExperimentAnalysis,
    HPRecommendation,
    ExperimentStatus,
    Metric, // Ensure Metric is imported
  } from "$lib/types";
  import {
    X,
    Clock,
    Tag,
    Settings,
    Pencil,
    Info,
    ChartLine,
    Eye,
    EyeClosed,
    Sparkle,
    ClipboardCheck,
    Copy,
    ChevronDown,
    CircleDot,
    Minimize2,
    Table2,    // Add this
    Loader2,   // Add this
  } from "lucide-svelte";
  import InteractiveChart from "./interactive-chart.svelte";
  import { page } from "$app/state";

  let {
    experiment = $bindable(),
    selectedExperiment = $bindable(),
    highlighted = $bindable(),
    selectedForDelete = $bindable(),
    selectedForEdit = $bindable(),
  }: {
    experiment: Experiment;
    selectedExperiment: Experiment | null;
    highlighted: string[];
    selectedForDelete: Experiment | null;
    selectedForEdit: Experiment | null;
  } = $props();

  let recommendations = $state<Record<string, HPRecommendation>>({});
  let activeRecommendation = $state<string | null>(null);
  let idCopied = $state<boolean>(false);
  let copiedParamKey = $state<string | null>(null);

  let showMetricsTable = $state(false);
  let rawMetrics = $state<Metric[]>([]);
  let metricsLoading = $state(false);
  let metricsError = $state<string | null>(null);

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
    OTHER: "Other Status", // Changed from "Other" for clarity
  };

  let currentStatusColor = $derived(
    experiment.status ? statusColors[experiment.status] : "bg-ctp-subtext0",
  );
  let currentStatusTooltip = $derived(
    experiment.status ? statusTooltips[experiment.status] : "Unknown Status",
  );

  async function toggleMetricsDisplay() {
    if (showMetricsTable) {
      // Currently showing table, switch to chart view
      showMetricsTable = false;
    } else {
      // Currently showing chart, switch to table view
      // Fetch data only if it hasn't been fetched yet, or if there was a previous error and no data currently displayed
      if (rawMetrics.length === 0 || (metricsError && rawMetrics.length === 0)) {
        metricsLoading = true;
        metricsError = null;
        try {
          const response = await fetch(`/api/experiments/${experiment.id}/metrics`);
          if (!response.ok) {
            throw new Error(`Failed to fetch metrics: ${response.statusText}`);
          }
          const data = await response.json();
          rawMetrics = data as Metric[]; // Assuming API returns Metric[]
          if (rawMetrics.length === 0) {
            metricsError = "No raw metric data points found for this experiment.";
          }
        } catch (err) {
          if (err instanceof Error) {
            metricsError = err.message;
          } else {
            metricsError = "An unknown error occurred while fetching metrics.";
          }
          rawMetrics = []; // Clear any previous data in case of error
        } finally {
          metricsLoading = false;
        }
      }
      showMetricsTable = true; // Show the table section (which will handle loader/error/data display internally)
    }
  }
</script>

<article class="h-full">
  <!-- Header with actions -->
  <header class="px-3 sm:px-4 py-3 bg-ctp-mantle border-b border-ctp-surface0">
    <!-- Mobile header -->
    <div class="flex flex-col sm:hidden w-full gap-2">
      <!-- Title row -->
      <h2 class="truncate">
        <span
          role="button"
          tabindex="0"
          class="text-base font-medium cursor-pointer transition-all duration-150 flex items-center gap-1.5"
          class:text-ctp-green={idCopied}
          class:text-ctp-text={!idCopied}
          onclick={() => {
            navigator.clipboard.writeText(experiment.id);
            idCopied = true;
            setTimeout(() => {
              idCopied = false;
            }, 800);
          }}
          onkeydown={(e) => {
            if (e.key === "Enter" || e.key === " ") e.currentTarget.click();
          }}
          title="Click to copy ID"
        >
          {#if idCopied}
            <span class="flex items-center">
              <ClipboardCheck size={16} class="mr-1 animate-bounce" />
              ID Copied
            </span>
          {:else}
            <span class="flex items-center">
              {experiment.name}
              <Copy size={12} class="ml-1 opacity-30 flex-shrink-0" />
            </span>
          {/if}
        </span>
      </h2>

      <!-- Actions row -->
      <div class="flex items-center justify-end gap-2">
        {#if page.data.user && page.data.user.id === experiment.user_id}
          <button
            class="p-1.5 text-ctp-subtext0 hover:text-ctp-text transition-transform active:rotate-90"
            onclick={async () => {
              const response = await fetch(
                `/api/ai/analysis?experimentId=${experiment.id}`,
              );
              const data = (await response.json()) as ExperimentAnalysis;
              recommendations = data.hyperparameter_recommendations;
            }}
            title="Get AI recommendations"
          >
            <Sparkle size={16} />
          </button>
          <button
            onclick={() => {
              selectedForEdit = experiment;
            }}
            class="p-1.5 text-ctp-subtext0 hover:text-ctp-text"
            title="Edit experiment"
          >
            <Pencil size={16} />
          </button>
        {/if}
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
                const uniqueIds = [...new Set([...data, experiment.id])];
                highlighted = uniqueIds;
              } catch (err) {}
            }
          }}
          class="p-1.5 text-ctp-subtext0 hover:text-ctp-text"
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
            class="p-1.5 text-ctp-subtext0 hover:text-ctp-red"
            aria-label="Delete"
            title="Delete experiment"
            onclick={(e) => {
              e.stopPropagation();
              selectedForDelete = experiment;
            }}
          >
            <X size={16} />
          </button>
        {/if}
        <button
          class="p-1.5 text-ctp-subtext0 hover:text-ctp-text"
          onclick={() => {
            selectedExperiment = null;
          }}
        >
          <Minimize2 size={16} />
        </button>
      </div>
    </div>

    <!-- Desktop/Tablet header -->
    <div class="hidden sm:flex sm:flex-row justify-between items-center">
      <h2 class="max-w-[70%]">
        <span
          role="button"
          tabindex="0"
          class="text-lg font-medium cursor-pointer transition-all duration-150 flex items-center gap-1.5"
          class:text-ctp-green={idCopied}
          class:text-ctp-text={!idCopied}
          onclick={() => {
            navigator.clipboard.writeText(experiment.id);
            idCopied = true;
            setTimeout(() => {
              idCopied = false;
            }, 800);
          }}
          onkeydown={(e) => {
            if (e.key === "Enter" || e.key === " ") e.currentTarget.click();
          }}
          title="Click to copy ID"
        >
          {#if idCopied}
            <span class="flex items-center">
              <ClipboardCheck size={18} class="mr-1 animate-bounce" />
              ID Copied
            </span>
          {:else}
            <span class="flex items-center truncate">
              <span class="truncate">{experiment.name}</span>
              <Copy size={14} class="ml-1 opacity-30 flex-shrink-0" />
            </span>
          {/if}
        </span>
      </h2>
      <div class="flex items-center gap-2">
        {#if page.data.user && page.data.user.id === experiment.user_id}
          <button
            class="p-1.5 text-ctp-subtext0 hover:text-ctp-text transition-transform active:rotate-90"
            onclick={async () => {
              const response = await fetch(
                `/api/ai/analysis?experimentId=${experiment.id}`,
              );
              const data = (await response.json()) as ExperimentAnalysis;
              recommendations = data.hyperparameter_recommendations;
            }}
            title="Get AI recommendations"
          >
            <Sparkle size={16} />
          </button>
          <button
            onclick={() => {
              selectedForEdit = experiment;
            }}
            class="p-1.5 text-ctp-subtext0 hover:text-ctp-text"
            title="Edit experiment"
          >
            <Pencil size={16} />
          </button>
        {/if}
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
                const uniqueIds = [...new Set([...data, experiment.id])];
                highlighted = uniqueIds;
              } catch (err) {}
            }
          }}
          class="p-1.5 text-ctp-subtext0 hover:text-ctp-text"
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
            class="p-1.5 text-ctp-subtext0 hover:text-ctp-red"
            aria-label="Delete"
            title="Delete experiment"
            onclick={(e) => {
              e.stopPropagation();
              selectedForDelete = experiment;
            }}
          >
            <X size={16} />
          </button>
        {/if}
        <button
          class="p-1.5 text-ctp-subtext0 hover:text-ctp-text"
          onclick={() => {
            selectedExperiment = null;
          }}
        >
          <Minimize2 size={16} />
        </button>
      </div>
    </div>
  </header>

  <!-- Content Area -->
  <div class="px-2 sm:px-4 py-3 flex flex-col gap-3">
    <!-- Metadata section -->
    <div
      class="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-4 text-ctp-subtext0 text-xs"
    >
      <div class="flex items-center gap-1">
        <Clock size={14} class="flex-shrink-0" />
        <time>
          {new Date(experiment.createdAt).toLocaleDateString("en-US", {
            year: "numeric",
            month: "short",
            day: "numeric",
          })}
        </time>
      </div>

      <!-- Experiment Status Display -->
      {#if experiment.status}
        <div class="flex items-center gap-1">
          <CircleDot size={14} class="flex-shrink-0 {currentStatusColor.replace('bg-', 'text-')}" /> {/* Use text color for icon */}
          <span class="font-medium {currentStatusColor.replace('bg-', 'text-')}">{currentStatusTooltip}</span>
        </div>
      {/if}

      {#if experiment.tags && experiment.tags.length > 0}
        <div
          class="flex items-center gap-1 overflow-x-auto sm:flex-wrap pb-1 sm:pb-0"
        >
          <Tag size={14} class="flex-shrink-0" />
          <div class="flex gap-1 flex-nowrap sm:flex-wrap">
            {#each experiment.tags as tag}
              <span
                class="whitespace-nowrap inline-flex items-center px-1.5 py-0.5 text-xs bg-ctp-surface0/50 text-ctp-blue rounded-full"
              >
                {tag}
              </span>
            {/each}
          </div>
        </div>
      {/if}
    </div>
    {#if experiment.description}
      <p
        class="
          text-ctp-text
          text-xs sm:text-sm
          leading-relaxed
          border-l-2 border-ctp-mauve
          pl-3 py-1.5 mt-1
          break-words
          sm:break-normal
        "
      >
        {experiment.description}
      </p>
    {/if}
    <!-- Parameters section -->
    {#if experiment.hyperparams && experiment.hyperparams.length > 0}
      <details class="mt-2 group">
        <summary
          class="flex items-center gap-2 cursor-pointer text-ctp-subtext0 hover:text-ctp-text py-1.5"
        >
          <Settings size={16} class="text-ctp-mauve flex-shrink-0" />
          <span class="text-sm font-medium">Hyperparameters</span>
          <ChevronDown
            size={16}
            class="ml-auto text-ctp-subtext0 group-open:rotate-180"
          />
        </summary>
        <div class="pt-2">
          <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
            {#each experiment.hyperparams as param (param.key)}
              <div class="flex items-center bg-ctp-mantle p-2 rounded-md overflow-hidden gap-1.5">
                <span class="text-xs font-medium text-ctp-subtext1 truncate shrink" title={param.key}>{param.key}</span>
                <span class="ml-auto text-xs text-ctp-text px-2 py-0.5 bg-ctp-surface0 rounded-sm truncate shrink" title={String(param.value)}>{param.value}</span>
                <div class="flex items-center flex-shrink-0">
                    {#if recommendations && recommendations[param.key]}
                      <button
                        class="p-0.5 rounded-sm text-ctp-subtext0 hover:text-ctp-lavender"
                        onclick={() => { activeRecommendation = recommendations[param.key].recommendation; }}
                        aria-label="Show recommendation"
                        title="Show AI recommendation"
                      >
                        <Info size={14} />
                      </button>
                    {/if}
                    <button
                      class="p-0.5 rounded-sm text-ctp-subtext0 hover:text-ctp-blue"
                      title="Copy {param.key}: {param.value}"
                      aria-label="Copy hyperparameter {param.key}"
                      onclick={() => {
                        navigator.clipboard.writeText(`${param.key}: ${param.value}`);
                        copiedParamKey = param.key;
                        setTimeout(() => {
                          if (copiedParamKey === param.key) {
                            copiedParamKey = null;
                          }
                        }, 1500); // Reset after 1.5 seconds
                      }}
                    >
                      {#if copiedParamKey === param.key}
                        <ClipboardCheck size={14} class="text-ctp-green" />
                      {:else}
                        <Copy size={14} />
                      {/if}
                    </button>
                </div>
              </div>
            {/each}
          </div>

          {#if activeRecommendation}
            <div
              class="mt-3 p-3 bg-ctp-surface0/50 border border-ctp-lavender/30 rounded-md relative"
            >
              <button
                class="absolute top-1.5 right-1.5 text-ctp-subtext0 hover:text-ctp-text"
                onclick={() => (activeRecommendation = null)}
                aria-label="Close recommendation"
              >
                <X size={14} />
              </button>
              <h4 class="text-xs font-medium text-ctp-lavender mb-1.5">
                AI Recommendation
              </h4>
              <p class="text-xs text-ctp-text leading-relaxed">
                {activeRecommendation}
              </p>
            </div>
          {/if}
        </div>
      </details>
    {/if}

    <!-- Metrics section -->
    {#if experiment.availableMetrics && experiment.availableMetrics.length > 0}
      <details class="mt-1 group">
        <summary class="flex items-center gap-2 cursor-pointer text-ctp-subtext0 hover:text-ctp-text py-1.5">
          <ChartLine size={16} class="text-ctp-blue" />
          <span class="text-sm font-medium">Metrics</span>
          <ChevronDown size={16} class="ml-auto text-ctp-subtext0 group-open:rotate-180" />
        </summary>
        <div class="pt-2">
          <!-- Toggle Button -->
          <div class="mb-3 text-right">
            <button
              class="inline-flex items-center gap-1.5 text-xs px-2 py-1 rounded-md text-ctp-subtext0 hover:text-ctp-text hover:bg-ctp-surface0 transition-colors"
              onclick={toggleMetricsDisplay}
              disabled={metricsLoading && !showMetricsTable} /* Disable only when initially loading table */
            >
              {#if showMetricsTable}
                <ChartLine size={14} /> Show Chart
              {:else}
                <Table2 size={14} /> Show Raw Data Table
              {/if}
              {#if metricsLoading && !showMetricsTable} <!-- Show loader on button only when initially loading table -->
                <Loader2 size={14} class="animate-spin ml-1" />
              {/if}
            </button>
          </div>

          <!-- Conditional Display: Chart or Table -->
          {#if showMetricsTable}
            {#if metricsLoading}
              <div class="flex justify-center items-center p-4 min-h-[100px]">
                <Loader2 size={20} class="animate-spin text-ctp-subtext0" />
                <span class="ml-2 text-ctp-subtext0 text-sm">Loading metrics...</span>
              </div>
            {:else if metricsError}
              <p class="text-xs text-ctp-red bg-ctp-red/10 p-3 rounded-md">{metricsError}</p>
            {:else if rawMetrics.length > 0}
              <div class="overflow-x-auto max-h-96 border border-ctp-surface0 rounded-md bg-ctp-base">
                <table class="w-full text-xs text-left">
                  <thead class="bg-ctp-mantle sticky top-0 z-10">
                    <tr>
                      <th class="p-2 font-medium text-ctp-subtext1">Name</th>
                      <th class="p-2 font-medium text-ctp-subtext1">Value</th>
                      <th class="p-2 font-medium text-ctp-subtext1">Step</th>
                      <th class="p-2 font-medium text-ctp-subtext1">Timestamp</th>
                    </tr>
                  </thead>
                  <tbody>
                    {#each rawMetrics as metric (metric.id)}
                      <tr class="border-t border-ctp-surface0 hover:bg-ctp-surface0/50">
                        <td class="p-2 text-ctp-text truncate max-w-xs" title={metric.name}>{metric.name}</td>
                        <td class="p-2 text-ctp-text" title={String(metric.value)}>{typeof metric.value === 'number' ? metric.value.toFixed(4) : metric.value}</td>
                        <td class="p-2 text-ctp-text">{metric.step ?? '-'}</td>
                        <td class="p-2 text-ctp-text whitespace-nowrap">
                          {new Date(metric.created_at).toLocaleString([], { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false })}
                        </td>
                      </tr>
                    {/each}
                  </tbody>
                </table>
              </div>
            {:else}
              <p class="text-xs text-ctp-overlay1 p-3 rounded-md">No metric data available to display in table. This might also indicate an issue if metrics were expected.</p>
            {/if}
          {:else}
            <!-- Original Chart Display -->
            <div class="-mx-2 sm:-mx-4">
              <div class="px-1 sm:px-2 w-full overflow-x-auto">
                <InteractiveChart {experiment} />
              </div>
            </div>
          {/if}
        </div>
      </details>
    {/if}
  </div>
</article>
