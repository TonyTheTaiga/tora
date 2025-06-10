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
  let idCopyAnimated = $state<boolean>(false); // Added this line
  let copiedParamKey = $state<string | null>(null);

  let showMetricsTable = $state(false);
  let rawMetrics = $state<Metric[]>([]);
  let metricsLoading = $state(false);
  let metricsError = $state<string | null>(null);

  // For Tag display
  let allTagsShown = $state(false);
  const initialTagLimit = 7;

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

<article class="h-full bg-ctp-mantle rounded-xl flex flex-col overflow-hidden">
  <!-- Header with actions -->
  <header class="px-4 sm:px-6 py-3 border-b border-ctp-surface1">
    <!-- Combined Header for both Mobile and Desktop -->
    <div class="flex items-center justify-end">
      <!-- Action Buttons -->
      <div
        class="flex items-center gap-1 bg-ctp-surface0/80 backdrop-blur-sm border border-ctp-surface1/50 rounded-full p-1 flex-shrink-0"
      >
        {#if page.data.user && page.data.user.id === experiment.user_id}
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
        {#if page.data.user && page.data.user.id === experiment.user_id}
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
      </div>
    </div>
  </header>

  <!-- Content Area -->
  <div class="px-4 sm:px-6 py-4 flex flex-col gap-4 overflow-y-auto flex-grow">
    <!-- Title, ID, and metadata -->
    <div class="flex flex-col gap-1 min-w-0 flex-grow mb-2">
      <h2
        class="text-xl sm:text-2xl font-bold text-ctp-text mb-1"
        title={experiment.name}
      >
        {experiment.name}
      </h2>
      <div class="flex flex-wrap items-center gap-4">
        <button
          type="button"
          aria-label="Copy Experiment ID"
          title={idCopied ? "ID Copied!" : "Copy Experiment ID"}
          class="flex items-center p-1 rounded-md text-ctp-subtext1 hover:bg-ctp-surface0 hover:text-ctp-text group flex-shrink-0 w-fit"
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
            <span class="text-xs text-ctp-green ml-1 hidden sm:inline"
              >Copied!</span
            >
          {:else}
            <Copy size={14} />
            <span
              class="text-xs text-ctp-subtext1 ml-1 hidden sm:inline group-hover:text-ctp-text transition-colors"
              >{experiment.id.substring(0, 8)}...</span
            >
          {/if}
        </button>

        <!-- Status and metadata row -->
        <div
          class="flex flex-wrap items-center gap-x-4 gap-y-2 text-ctp-subtext0 text-sm"
        >
          <div class="flex items-center gap-1.5">
            <Clock size={15} class="flex-shrink-0 text-ctp-overlay1" />
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

          <div
            class="flex items-center gap-1.5 p-1 px-2 rounded-full {experiment.visibility ===
            'PUBLIC'
              ? 'text-ctp-green bg-ctp-green/20'
              : 'text-ctp-red bg-ctp-red/20'}"
            title={experiment.visibility === "PUBLIC" ? "Public" : "Private"}
          >
            {#if experiment.visibility === "PUBLIC"}
              <Globe size={15} />
              <span class="text-xs text-ctp-green">Public</span>
            {:else}
              <GlobeLock size={15} />
              <span class="text-xs text-ctp-red">Private</span>
            {/if}
          </div>
        </div>
      </div>
    </div>

    <!-- Metadata section -->
    {#if experiment.tags && experiment.tags.length > 0}
      <div class="flex items-start gap-1.5 text-ctp-subtext0 text-sm">
        <Tag size={15} class="flex-shrink-0 text-ctp-overlay1 mt-0.5" />
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
        class="
          text-ctp-subtext0
          text-xs sm:text-sm
          leading-relaxed
          border-l-2 border-ctp-mauve
          pl-4 py-2 mt-2
          break-words
          sm:break-normal
          description-truncate-detailed
          mb-4
        "
        title={experiment.description}
      >
        {experiment.description}
      </p>
    {/if}
    <!-- Parameters section -->
    {#if experiment.hyperparams && experiment.hyperparams.length > 0}
      <details
        class="mt-2 group open rounded-lg bg-ctp-surface0/80 backdrop-blur-sm border border-ctp-surface1/50"
        open
      >
        <summary
          class="flex items-center gap-3 px-3 py-2 rounded-t-lg cursor-pointer"
        >
          <div class="flex items-center gap-2 flex-grow">
            <Settings size={18} class="text-ctp-blue" />
            <span class="text-base font-semibold text-ctp-text"
              >Hyperparameters</span
            >
          </div>
          <span
            class="bg-ctp-blue/20 text-ctp-blue border border-ctp-blue/40 rounded-md px-2 py-0.5 text-xs shrink-0"
          >
            {experiment.hyperparams?.length || 0} params
          </span>
          <ChevronDown
            size={18}
            class="text-ctp-subtext0 group-open:rotate-180 transition-transform shrink-0"
          />
        </summary>
        <div class="pt-2 px-3 pb-3 space-y-2">
          {#each visibleHyperparameters as param (param.key)}
            <div
              class="flex items-center justify-between p-2 rounded-lg bg-ctp-surface1/60 hover:bg-ctp-surface1/80 group backdrop-blur-sm border border-ctp-surface2/30"
            >
              <div class="flex items-center space-x-3 flex-1 min-w-0">
                <span
                  class="text-ctp-subtext1 font-medium truncate shrink"
                  title={param.key}>{param.key}</span
                >
                {#if recommendations && recommendations[param.key]}
                  <button
                    class="p-0.5 rounded-sm text-ctp-overlay2 hover:text-ctp-lavender hover:bg-ctp-surface2 flex-shrink-0"
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

              <!-- Right Part: Value + Copy Button -->
              <div class="flex items-center space-x-2">
                <code
                  ><pre
                    class="text-ctp-text font-mono bg-ctp-surface2/80 px-2 py-1 rounded text-sm truncate max-w-[150px] sm:max-w-xs backdrop-blur-sm border border-ctp-overlay0/20">{param.value}</pre></code
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
              class="w-full text-ctp-subtext0 hover:text-ctp-text mt-4 rounded-lg py-2.5 px-4 text-sm flex items-center justify-center bg-ctp-surface1/60 hover:bg-ctp-surface1/80 backdrop-blur-sm border border-ctp-surface2/30"
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
        class="mt-2 group open rounded-lg bg-ctp-surface0/80 backdrop-blur-sm border border-ctp-surface1/50"
        open
      >
        <summary
          class="flex items-center gap-3 px-3 py-2 rounded-t-lg cursor-pointer"
        >
          <div class="flex items-center gap-2 flex-grow">
            <ChartArea size={18} class="text-ctp-blue" />
            <span class="text-base font-semibold text-ctp-text">Metrics</span>
          </div>
          <ChevronDown
            size={18}
            class="text-ctp-subtext0 group-open:rotate-180 transition-transform shrink-0"
          />
        </summary>
        <div class="pt-2 px-3 pb-3 space-y-3">
          <!-- New Segmented Control -->
          <div class="flex justify-center mb-4">
            <div
              class="flex bg-ctp-surface1/80 w-full rounded-lg p-1 space-x-1 backdrop-blur-sm border border-ctp-surface2/30"
            >
              <button
                class="flex-1 px-3 py-1.5 rounded-md text-sm font-medium flex items-center justify-center gap-2 focus:outline-none {!showMetricsTable
                  ? 'bg-ctp-surface2/80 text-ctp-text backdrop-blur-sm'
                  : 'text-ctp-subtext0 hover:bg-ctp-surface2/40'}"
                onclick={() => {
                  showMetricsTable = false;
                }}
              >
                <ChartLine size={16} />
                Chart
              </button>
              <button
                class="flex-1 px-3 py-1.5 rounded-md text-sm font-medium flex items-center justify-center gap-2 focus:outline-none {showMetricsTable
                  ? 'bg-ctp-surface2/80 text-ctp-text backdrop-blur-sm'
                  : 'text-ctp-subtext0 hover:bg-ctp-surface2/40'}"
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

          <!-- Conditional Display: Chart or Table -->
          {#if showMetricsTable}
            {#if metricsLoading}
              <div
                class="flex flex-col justify-center items-center p-6 min-h-[150px] bg-ctp-surface0/60 rounded-lg text-center backdrop-blur-sm border border-ctp-surface1/40"
              >
                <Loader2
                  size={28}
                  class="animate-spin text-ctp-subtext0 mb-3"
                />
                <span class="text-ctp-subtext0 text-base"
                  >Loading metrics...</span
                >
                <p class="text-ctp-overlay1 text-xs mt-1">
                  Please wait a moment.
                </p>
              </div>
            {:else if metricsError}
              <p class="text-sm text-red-300 bg-red-900/20 p-4 rounded-lg">
                {metricsError}
              </p>
            {:else if rawMetrics.length > 0}
              <div
                class="overflow-x-auto max-h-[500px] bg-ctp-surface0/60 rounded-lg p-4 backdrop-blur-sm border border-ctp-surface1/40"
              >
                <table class="w-full text-sm text-left">
                  <thead
                    class="bg-ctp-surface1/80 sticky top-0 z-10 backdrop-blur-sm"
                  >
                    <tr>
                      <th class="p-3 font-semibold text-ctp-text">Name</th>
                      <th class="p-3 font-semibold text-ctp-text">Value</th>
                      <th class="p-3 font-semibold text-ctp-text">Step</th>
                      <th class="p-3 font-semibold text-ctp-text">Timestamp</th>
                    </tr>
                  </thead>
                  <tbody class="divide-y divide-ctp-surface1">
                    {#each rawMetrics as metric (metric.id)}
                      <tr class="">
                        <td
                          class="p-3 text-ctp-subtext1 truncate max-w-sm"
                          title={metric.name}>{metric.name}</td
                        >
                        <td
                          class="p-3 text-ctp-text truncate max-w-sm"
                          title={String(metric.value)}
                          >{typeof metric.value === "number"
                            ? metric.value.toFixed(4)
                            : metric.value}</td
                        >
                        <td class="p-3 text-ctp-subtext1 truncate max-w-[70px]"
                          >{metric.step ?? "N/A"}</td
                        >
                        <td class="p-3 text-ctp-subtext1 whitespace-nowrap">
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
                class="text-sm text-ctp-overlay2 bg-ctp-surface0/60 p-4 rounded-lg text-center backdrop-blur-sm border border-ctp-surface1/40"
              >
                No metric data points found for this experiment.
              </p>
            {/if}
          {:else}
            <div
              class="bg-ctp-surface0/60 rounded-lg p-4 backdrop-blur-sm border border-ctp-surface1/40"
            >
              <InteractiveChart {experiment} />
            </div>
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
      /* white-space: normal; */ /* Ensure it wraps */
    }
  }
</style>
