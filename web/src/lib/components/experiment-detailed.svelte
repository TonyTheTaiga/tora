<script lang="ts">
  import type {
    Experiment,
    ExperimentAnalysis,
    HPRecommendation,
    Metric,
  } from "$lib/types";

  interface ExperimentWithMetrics extends Experiment {
    metricData?: Record<string, number[]>;
  }
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
    Minimize2,
    Table2,
    Loader2,
    Globe,
    GlobeLock,
    ChartArea,
  } from "lucide-svelte";
  import InteractiveChart from "./interactive-chart.svelte";
  import { page } from "$app/state";
  import {
    openDeleteExperimentModal,
    openEditExperimentModal,
    setSelectedExperiment,
  } from "$lib/state/app.svelte.js";

  let {
    experiment = $bindable(),
    highlighted = $bindable(),
  }: {
    experiment: ExperimentWithMetrics;
    highlighted: string[];
  } = $props();

  let recommendations = $state<Record<string, HPRecommendation>>({});
  let activeRecommendation = $state<string | null>(null);
  let idCopied = $state<boolean>(false);
  let idCopyAnimated = $state<boolean>(false);
  let copiedParamKey = $state<string | null>(null);
  let showMetricsTable = $state(false);
  let rawMetrics = $state<Metric[]>([]);
  let metricsLoading = $state(false);
  let metricsError = $state<string | null>(null);
  let allTagsShown = $state(false);
  const initialTagLimit = 7;

  let currentWorkspace = $derived(page.data.currentWorkspace);
  let canEditExperiment = $derived(
    page.data.user &&
      (page.data.user.id === experiment.user_id ||
        (currentWorkspace &&
          ["OWNER", "ADMIN", "EDITOR"].includes(currentWorkspace.role))),
  );
  let canDeleteExperiment = $derived(
    page.data.user &&
      (page.data.user.id === experiment.user_id ||
        (currentWorkspace &&
          ["OWNER", "ADMIN"].includes(currentWorkspace.role))),
  );

  let visibleTags = $derived.by(() => {
    if (!experiment.tags || !Array.isArray(experiment.tags)) return [];
    if (allTagsShown || experiment.tags.length <= initialTagLimit) {
      return experiment.tags;
    }
    return experiment.tags.slice(0, initialTagLimit);
  });

  let hiddenTagCount = $derived.by(() => {
    if (
      !experiment.tags ||
      !Array.isArray(experiment.tags) ||
      allTagsShown ||
      experiment.tags.length <= initialTagLimit
    ) {
      return 0;
    }
    return experiment.tags.length - initialTagLimit;
  });

  function showAllTags() {
    allTagsShown = true;
  }
  function showLessTags() {
    allTagsShown = false;
  }

  let allHyperparametersShown = $state(false);
  const initialHyperparameterLimit = 7;

  let visibleHyperparameters = $derived.by(() => {
    const hps = experiment.hyperparams || [];
    if (!Array.isArray(hps)) return [];
    if (allHyperparametersShown || hps.length <= initialHyperparameterLimit) {
      return hps;
    }
    return hps.slice(0, initialHyperparameterLimit);
  });

  let hiddenHyperparameterCount = $derived.by(() => {
    const hps = experiment.hyperparams || [];
    if (
      !Array.isArray(hps) ||
      allHyperparametersShown ||
      hps.length <= initialHyperparameterLimit
    ) {
      return 0;
    }
    return hps.length - initialHyperparameterLimit;
  });

  let availableMetrics = $derived.by(() =>
    experiment.metricData
      ? Object.keys(experiment.metricData)
      : experiment.availableMetrics || [],
  );

  async function fetchRawMetricsIfNeeded() {
    // Fetch data only if it hasn't been fetched yet, or if there was a previous error and no data currently displayed
    if (rawMetrics.length === 0 || (metricsError && rawMetrics.length === 0)) {
      metricsLoading = true;
      metricsError = null;
      try {
        const response = await fetch(
          `/api/experiments/${experiment.id}/metrics`,
        );
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
  }
</script>

<header>
  <div class="flex items-center justify-end">
    <div
      class="flex items-center gap-1 bg-gradient-to-b from-bg-ctp-surface0/80 to-bg-ctp-mantle backdrop-blur-sm border-t border-x border-ctp-surface1/50 rounded-t-xl p-1"
    >
      {#if canEditExperiment}
        <button
          class="p-1.5 rounded-full text-ctp-subtext0 hover:text-ctp-lavender hover:bg-ctp-surface1/60 transition-colors"
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
            openEditExperimentModal(experiment);
          }}
          class="p-1.5 rounded-full text-ctp-subtext0 hover:text-ctp-blue hover:bg-ctp-surface1/60 transition-colors"
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
        class="p-1.5 rounded-full text-ctp-subtext0 hover:text-ctp-teal hover:bg-ctp-surface1/60 transition-colors"
        title="Show experiment chain"
      >
        {#if highlighted.includes(experiment.id)}
          <EyeClosed size={16} />
        {:else}
          <Eye size={16} />
        {/if}
      </button>
      {#if canDeleteExperiment}
        <button
          type="button"
          class="p-1.5 rounded-full text-ctp-subtext0 hover:text-ctp-red hover:bg-ctp-surface1/60 transition-colors"
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
      <button
        class="p-1.5 rounded-full text-ctp-subtext0 hover:text-ctp-text hover:bg-ctp-surface1/60 transition-colors"
        onclick={() => {
          setSelectedExperiment(null);
        }}
        title="Minimize"
      >
        <Minimize2 size={16} />
      </button>
      <div
        class="p-1.5 rounded-full text-ctp-subtext0"
        title={experiment.visibility === "PUBLIC" ? "Public" : "Private"}
      >
        {#if experiment.visibility === "PUBLIC"}
          <Globe size={16} class="text-ctp-green" />
        {:else}
          <GlobeLock size={16} class="text-ctp-red" />
        {/if}
      </div>
    </div>
  </div>
</header>

<article
  class="h-full bg-ctp-mantle rounded-b-xl flex flex-col overflow-hidden"
>
  <!-- Header with actions -->
  <!-- Content Area -->
  <div class="px-4 sm:px-6 py-4 flex flex-col gap-4 overflow-y-auto flex-grow">
    <!-- Date and Title -->
    <div class="flex flex-col gap-1 min-w-0 mb-2 overflow-hidden">
      <div class="flex items-center gap-1.5 text-ctp-subtext0 text-xs">
        <Clock size={14} class="text-ctp-overlay1" />
        <time class="text-ctp-subtext1">
          {new Date(experiment.createdAt).toLocaleString("en-US", {
            year: "numeric",
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
          })}
        </time>
      </div>
      <h2
        class="text-lg sm:text-xl md:text-2xl font-semibold text-ctp-text mb-1 leading-tight break-words overflow-wrap-anywhere"
        title={experiment.name}
      >
        {experiment.name}
      </h2>
      <button
        type="button"
        aria-label="Copy Experiment ID"
        title={idCopied ? "ID Copied!" : "Copy Experiment ID"}
        class="flex items-center p-1 rounded-md text-ctp-subtext1 hover:bg-ctp-surface0 hover:text-ctp-text group w-fit"
        onclick={() => {
          navigator.clipboard.writeText(experiment.id);
          idCopied = true;
          idCopyAnimated = true;
          setTimeout(() => {
            idCopied = false;
          }, 1200);
          setTimeout(() => {
            idCopyAnimated = false;
          }, 400);
        }}
      >
        {#if idCopied}
          <ClipboardCheck
            size={14}
            class="text-ctp-green transition-transform duration-200 {idCopyAnimated
              ? 'scale-125'
              : ''}"
          />
          <span class="text-xs text-ctp-green ml-1">Copied!</span>
        {:else}
          <Copy size={14} />
          <span
            class="text-xs text-ctp-subtext1 ml-1 group-hover:text-ctp-text transition-colors"
            >{experiment.id.substring(0, 8)}...</span
          >
        {/if}
      </button>
    </div>

    <!-- Metadata section -->
    {#if experiment.tags && experiment.tags.length > 0}
      <div
        class="flex items-start gap-1.5 text-ctp-subtext0 text-xs sm:text-sm"
      >
        <Tag size={15} class="text-ctp-overlay1 mt-0.5" />
        <div class="flex flex-wrap gap-1.5 items-center">
          {#each visibleTags as tag}
            <span
              class="whitespace-nowrap inline-flex items-center px-2 py-1 text-xs text-ctp-blue rounded-full bg-ctp-blue/20 border border-ctp-blue/30 truncate max-w-[150px]"
              title={tag}
            >
              {tag}
            </span>
          {/each}
          {#if hiddenTagCount > 0}
            <button
              type="button"
              onclick={showAllTags}
              class="text-xs text-ctp-sky hover:text-ctp-blue hover:bg-ctp-surface0 px-2 py-0.5 rounded-md"
            >
              +{hiddenTagCount} more
            </button>
          {/if}
          {#if allTagsShown && experiment.tags.length > initialTagLimit}
            <button
              type="button"
              onclick={showLessTags}
              class="text-xs text-ctp-sky hover:text-ctp-blue hover:bg-ctp-surface0 px-2 py-0.5 rounded-md"
            >
              Show less
            </button>
          {/if}
        </div>
      </div>
    {/if}
    {#if experiment.description}
      <p
        class="text-ctp-subtext0 text-xs sm:text-sm leading-relaxed border-l-2 border-ctp-mauve pl-3 sm:pl-4 py-2 mt-2 break-words description-truncate-detailed mb-3 sm:mb-4"
        title={experiment.description}
      >
        {experiment.description}
      </p>
    {/if}
    <!-- Parameters section -->
    {#if experiment.hyperparams && experiment.hyperparams.length > 0}
      <details
        class="mt-2 group open rounded-lg bg-ctp-surface0/40 backdrop-blur-sm border border-ctp-surface1/30"
        open
      >
        <summary
          class="flex items-center gap-3 px-3 py-2 rounded-t-lg cursor-pointer"
        >
          <div class="flex items-center gap-2 flex-grow">
            <Settings size={18} class="text-ctp-blue" />
            <span class="text-sm sm:text-base font-semibold text-ctp-text"
              >Hyperparameters</span
            >
          </div>
          <span
            class="bg-ctp-blue/20 text-ctp-blue border border-ctp-blue/40 rounded-md px-2 py-0.5 text-xs"
          >
            {experiment.hyperparams?.length || 0} params
          </span>
          <ChevronDown
            size={18}
            class="text-ctp-subtext0 group-open:rotate-180 transition-transform"
          />
        </summary>
        <div class="pt-2 px-3 pb-3 space-y-2">
          {#each visibleHyperparameters as param (param.key)}
            <div
              class="flex items-center justify-between p-2 rounded-lg bg-ctp-surface1/30 hover:bg-ctp-surface1/50 group backdrop-blur-sm border border-ctp-surface2/20"
            >
              <div class="flex items-center space-x-3 flex-1 min-w-0">
                <span
                  class="text-ctp-subtext1 font-medium truncate text-xs sm:text-sm"
                  title={param.key}>{param.key}</span
                >
                {#if recommendations && recommendations[param.key]}
                  <button
                    class="p-0.5 rounded-sm text-ctp-overlay2 hover:text-ctp-lavender hover:bg-ctp-surface2"
                    onclick={() => {
                      activeRecommendation =
                        recommendations[param.key].recommendation;
                    }}
                    aria-label="Show recommendation"
                    title="Show AI recommendation"
                  >
                    <Info size={14} />
                  </button>
                {/if}
              </div>

              <div class="flex items-center space-x-2">
                <code
                  ><pre
                    class="text-ctp-text font-mono bg-ctp-surface2/80 px-2 py-1 rounded text-xs sm:text-sm truncate max-w-[120px] sm:max-w-[150px] md:max-w-xs backdrop-blur-sm border border-ctp-overlay0/20">{param.value}</pre></code
                >
                <button
                  type="button"
                  class="p-1 rounded text-ctp-subtext1 hover:bg-ctp-surface2/60 hover:text-ctp-text"
                  title="Copy value"
                  aria-label="Copy hyperparameter value {param.value}"
                  onclick={() => {
                    navigator.clipboard.writeText(
                      String(param.value), // Ensure value is string for clipboard
                    );
                    copiedParamKey = param.key;
                    setTimeout(() => {
                      if (copiedParamKey === param.key) {
                        copiedParamKey = null;
                      }
                    }, 1200);
                  }}
                >
                  {#if copiedParamKey === param.key}
                    <ClipboardCheck size={14} class="text-green-400" />
                  {:else}
                    <Copy size={14} />
                  {/if}
                </button>
              </div>
            </div>
          {/each}

          {#if (experiment.hyperparams?.length || 0) > initialHyperparameterLimit}
            <button
              type="button"
              onclick={() => {
                allHyperparametersShown = !allHyperparametersShown;
              }}
              class="w-full text-ctp-subtext0 hover:text-ctp-text mt-4 rounded-lg py-2 px-3 text-xs sm:text-sm flex items-center justify-center bg-ctp-surface1/30 hover:bg-ctp-surface1/50 backdrop-blur-sm border border-ctp-surface2/20"
            >
              <ChevronDown
                class="w-4 h-4 mr-2 transition-transform {allHyperparametersShown
                  ? 'rotate-180'
                  : ''}"
              />
              {allHyperparametersShown
                ? "Show less"
                : `Show +${hiddenHyperparameterCount} more hyperparameters`}
            </button>
          {/if}
          {#if activeRecommendation}
            <div
              class="mt-3 p-3.5 bg-ctp-surface1/80 rounded-lg relative backdrop-blur-sm border border-ctp-lavender/50"
            >
              <button
                class="absolute top-2 right-2 p-1 rounded-md text-ctp-subtext1 hover:text-ctp-text hover:bg-ctp-surface2"
                onclick={() => (activeRecommendation = null)}
                aria-label="Close recommendation"
              >
                <X size={15} />
              </button>
              <h4 class="text-sm font-semibold text-purple-300 mb-2">
                AI Recommendation
              </h4>
              <p class="text-sm text-white leading-relaxed">
                {activeRecommendation}
              </p>
            </div>
          {/if}
        </div>
      </details>
    {/if}

    <!-- Metrics section -->
    {#if availableMetrics.length > 0}
      <details
        class="mt-2 group open rounded-lg bg-ctp-surface0/40 backdrop-blur-sm border border-ctp-surface1/30"
        open
      >
        <summary
          class="flex items-center gap-3 px-3 py-2 rounded-t-lg cursor-pointer"
        >
          <div class="flex items-center gap-2 flex-grow">
            <ChartArea size={18} class="text-ctp-blue" />
            <span class="text-sm sm:text-base font-semibold text-ctp-text"
              >Metrics</span
            >
          </div>
          <ChevronDown
            size={18}
            class="text-ctp-subtext0 group-open:rotate-180 transition-transform"
          />
        </summary>
        <div class="pt-2 px-3 pb-3 space-y-3">
          <div class="flex justify-center mb-4">
            <div
              class="flex bg-ctp-surface1/40 w-full rounded-lg p-1 space-x-1 backdrop-blur-sm border border-ctp-surface2/20"
            >
              <button
                class="flex-1 px-2 sm:px-3 py-1.5 rounded-md text-xs sm:text-sm font-medium flex items-center justify-center gap-1 sm:gap-2 focus:outline-none {!showMetricsTable
                  ? 'bg-ctp-surface2/50 text-ctp-text backdrop-blur-sm'
                  : 'text-ctp-subtext0 hover:bg-ctp-surface2/30'}"
                onclick={() => {
                  showMetricsTable = false;
                }}
              >
                <ChartLine size={16} />
                Chart
              </button>
              <button
                class="flex-1 px-2 sm:px-3 py-1.5 rounded-md text-xs sm:text-sm font-medium flex items-center justify-center gap-1 sm:gap-2 focus:outline-none {showMetricsTable
                  ? 'bg-ctp-surface2/50 text-ctp-text backdrop-blur-sm'
                  : 'text-ctp-subtext0 hover:bg-ctp-surface2/30'}"
                onclick={() => {
                  showMetricsTable = true;
                  fetchRawMetricsIfNeeded();
                }}
                disabled={metricsLoading}
              >
                <Table2 size={16} />
                Data
              </button>
            </div>
          </div>

          {#if showMetricsTable}
            {#if metricsLoading}
              <div
                class="flex flex-col justify-center items-center p-6 min-h-[150px] bg-ctp-surface0/30 rounded-lg text-center backdrop-blur-sm border border-ctp-surface1/20"
              >
                <Loader2
                  size={28}
                  class="animate-spin text-ctp-subtext0 mb-3"
                />
                <span class="text-ctp-subtext0 text-sm sm:text-base"
                  >Loading metrics...</span
                >
                <p class="text-ctp-overlay1 text-xs sm:text-sm mt-1">
                  Please wait a moment.
                </p>
              </div>
            {:else if metricsError}
              <p
                class="text-xs sm:text-sm text-ctp-red bg-ctp-red/20 rounded-lg"
              >
                {metricsError}
              </p>
            {:else if rawMetrics.length > 0}
              <div
                class="overflow-x-auto max-h-[500px] bg-ctp-surface0/30 rounded-lg p-4 backdrop-blur-sm border border-ctp-surface1/20"
              >
                <table class="w-full text-sm text-left">
                  <thead
                    class="bg-ctp-surface1/40 sticky top-0 z-10 backdrop-blur-sm"
                  >
                    <tr>
                      <th
                        class="p-2 sm:p-3 font-semibold text-ctp-text text-xs sm:text-sm"
                        >Name</th
                      >
                      <th
                        class="p-2 sm:p-3 font-semibold text-ctp-text text-xs sm:text-sm"
                        >Value</th
                      >
                      <th
                        class="p-2 sm:p-3 font-semibold text-ctp-text text-xs sm:text-sm"
                        >Step</th
                      >
                      <th
                        class="p-2 sm:p-3 font-semibold text-ctp-text text-xs sm:text-sm hidden sm:table-cell"
                        >Timestamp</th
                      >
                    </tr>
                  </thead>
                  <tbody class="divide-y divide-ctp-surface1">
                    {#each rawMetrics as metric (metric.id)}
                      <tr>
                        <td
                          class="p-2 sm:p-3 text-ctp-subtext1 truncate max-w-[120px] sm:max-w-sm text-xs sm:text-sm"
                          title={metric.name}>{metric.name}</td
                        >
                        <td
                          class="p-2 sm:p-3 text-ctp-text truncate max-w-[100px] sm:max-w-sm text-xs sm:text-sm"
                          title={String(metric.value)}
                          >{typeof metric.value === "number"
                            ? metric.value.toFixed(4)
                            : metric.value}</td
                        >
                        <td
                          class="p-2 sm:p-3 text-ctp-subtext1 truncate max-w-[60px] sm:max-w-[70px] text-xs sm:text-sm"
                          >{metric.step ?? "N/A"}</td
                        >
                        <td
                          class="p-2 sm:p-3 text-ctp-subtext1 whitespace-nowrap text-xs sm:text-sm hidden sm:table-cell"
                        >
                          {new Date(metric.created_at).toLocaleString("en-US", {
                            year: "numeric",
                            month: "short",
                            day: "numeric",
                            hour: "2-digit",
                            minute: "2-digit",
                          })}
                        </td>
                      </tr>
                    {/each}
                  </tbody>
                </table>
              </div>
            {:else}
              <p
                class="text-xs sm:text-sm text-ctp-overlay2 bg-ctp-surface0/30 p-3 sm:p-4 rounded-lg text-center backdrop-blur-sm border border-ctp-surface1/20"
              >
                No metric data points found for this experiment.
              </p>
            {/if}
          {:else}
            <InteractiveChart {experiment} />
          {/if}
        </div>
      </details>
    {/if}
  </div>
</article>

<style>
  .description-truncate-detailed {
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: 5; /* Show 5 lines for detailed view */
    line-clamp: 5;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  /* Fallback for non-webkit browsers */
  @supports not (-webkit-line-clamp: 5) {
    .description-truncate-detailed {
      max-height: calc(
        1.5em * 5
      ); /* Assuming line-height ~1.5em, for 5 lines */
    }
  }
</style>
