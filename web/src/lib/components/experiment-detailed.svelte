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

  function showAllHyperparameters() {
    allHyperparametersShown = true;
  }
  function showLessHyperparameters() {
    allHyperparametersShown = false;
  }

  let availableMetrics = $derived.by(() =>
    experiment.metricData
      ? Object.keys(experiment.metricData)
      : experiment.availableMetrics || [],
  );

  async function fetchRawMetricsIfNeeded() {
    // Fetch data only if it hasn't been fetched yet, or if there was a previous error and no data currently displayed
    if (
      rawMetrics.length === 0 ||
      (metricsError && rawMetrics.length === 0)
    ) {
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
          metricsError =
            "No raw metric data points found for this experiment.";
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

<article
  class="h-full bg-ctp-crust rounded-xl shadow-lg flex flex-col overflow-hidden"
>
  <!-- Header with actions -->
  <header class="px-4 sm:px-6 py-5 bg-ctp-mantle border-b border-ctp-surface1"> {# py-4 to py-5 #}
    <!-- Combined Header for both Mobile and Desktop -->
    <div class="flex items-center justify-between">
      <!-- Action Buttons -->
      <div class="flex items-center gap-2 flex-shrink-0">
        {#if page.data.user && page.data.user.id === experiment.user_id}
          <button
            class="p-2 rounded-lg text-ctp-subtext0 hover:text-ctp-lavender hover:bg-ctp-surface0 transition-colors"
            onclick={async () => {
              const response = await fetch(
                `/api/ai/analysis?experimentId=${experiment.id}`,
              );
              const data = (await response.json()) as ExperimentAnalysis;
              recommendations = data.hyperparameter_recommendations;
            }}
            title="Get AI recommendations"
          >
            <Sparkle size={18} />
          </button>
          <button
            onclick={() => {
              openEditExperimentModal(experiment);
            }}
            class="p-2 rounded-lg text-ctp-subtext0 hover:text-ctp-blue hover:bg-ctp-surface0 transition-colors"
            title="Edit experiment"
          >
            <Pencil size={18} />
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
          class="p-2 rounded-lg text-ctp-subtext0 hover:text-ctp-teal hover:bg-ctp-surface0 transition-colors"
          title="Show experiment chain"
        >
          {#if highlighted.includes(experiment.id)}
            <EyeClosed size={18} />
          {:else}
            <Eye size={18} />
          {/if}
        </button>
        {#if page.data.user && page.data.user.id === experiment.user_id}
          <button
            type="button"
            class="p-2 rounded-lg text-ctp-subtext0 hover:text-ctp-red hover:bg-ctp-red/20 transition-colors"
            aria-label="Delete"
            title="Delete experiment"
            onclick={(e) => {
              e.stopPropagation();
              openDeleteExperimentModal(experiment);
            }}
          >
            <X size={18} />
          </button>
        {/if}
        <button
          class="p-2 rounded-lg text-ctp-subtext0 hover:text-ctp-text hover:bg-ctp-surface0 transition-colors"
          onclick={() => {
            setSelectedExperiment(null);
          }}
          title="Minimize"
        >
          <Minimize2 size={18} />
        </button>
      </div>

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
          class="flex items-center gap-1.5 p-1 px-2 rounded-full transition-colors {experiment.visibility ===
          'PUBLIC'
            ? 'text-ctp-green bg-ctp-green/10 hover:bg-ctp-green/20'
            : 'text-ctp-red bg-ctp-red/10 hover:bg-ctp-red/20'}"
          title={experiment.visibility === "PUBLIC" ? "Public" : "Private"}
        >
          {#if experiment.visibility === "PUBLIC"}
            <Globe size={15} />
            <span class="text-xs">Public</span>
          {:else}
            <GlobeLock size={15} />
            <span class="text-xs">Private</span>
          {/if}
        </div>
      </div>
    </div>
  </header>

  <!-- Content Area -->
  <div class="px-4 sm:px-6 py-5 flex flex-col gap-5 overflow-y-auto flex-grow">
    <!-- MOVED: Title and ID -->
    <div class="flex flex-col gap-1 min-w-0 flex-grow mb-3">
      <h2
        class="text-2xl sm:text-3xl font-bold text-ctp-text mb-2" {# Changed classes #}
        title={experiment.name}
      >
        {experiment.name}
      </h2>
      <button
        type="button"
        aria-label="Copy Experiment ID"
        title={idCopied ? "ID Copied!" : "Copy Experiment ID"}
        class="flex items-center p-1 rounded-md text-ctp-subtext1 transition-colors hover:bg-ctp-surface0 hover:text-ctp-text active:bg-ctp-surface1 group flex-shrink-0 w-fit"
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
    </div>

    <!-- Metadata section -->
    {#if experiment.tags && experiment.tags.length > 0}
      <div class="flex items-start gap-1.5 text-ctp-subtext0 text-sm">
        <Tag size={15} class="flex-shrink-0 text-ctp-overlay1 mt-0.5" />
        <div class="flex flex-wrap gap-1.5 items-center">
          {#each visibleTags as tag}
            <span
              class="whitespace-nowrap inline-flex items-center px-2 py-1 text-xs text-ctp-blue rounded-full bg-ctp-blue/20 border border-ctp-blue/30 truncate max-w-[150px]" {# Changed classes #}
              title={tag}
            >
              {tag}
            </span>
          {/each}
          {#if hiddenTagCount > 0}
            <button
              type="button"
              onclick={showAllTags}
              class="text-xs text-ctp-sky hover:text-ctp-blue hover:underline focus:outline-none hover:bg-ctp-surface0 px-2 py-0.5 rounded-md transition-colors"
            >
              +{hiddenTagCount} more
            </button>
          {/if}
          {#if allTagsShown && experiment.tags.length > initialTagLimit}
            <button
              type="button"
              onclick={showLessTags}
              class="text-xs text-ctp-sky hover:text-ctp-blue hover:underline focus:outline-none hover:bg-ctp-surface0 px-2 py-0.5 rounded-md transition-colors"
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
          text-ctp-subtext0 {# Changed text-ctp-text to text-ctp-subtext0 #}
          text-xs sm:text-sm
          leading-relaxed
          border-l-2 border-ctp-mauve
          pl-4 py-2 mt-2
          break-words
          sm:break-normal
          description-truncate-detailed
          mb-4 {# Added mb-4 #}
        "
        title={experiment.description}
      >
        {experiment.description}
      </p>
    {/if}
    <!-- Parameters section -->
    {#if experiment.hyperparams && experiment.hyperparams.length > 0}
      <details class="mt-2 group bg-ctp-mantle/60 border border-ctp-overlay0/30 backdrop-blur-sm rounded-lg" open>
        <summary
          class="flex items-center gap-2.5 cursor-pointer text-ctp-text px-4 py-3 hover:bg-ctp-overlay0/20 transition-colors rounded-t-lg"
        >
          <Settings size={18} class="text-ctp-overlay1 flex-shrink-0" />
          <span class="text-lg font-semibold text-ctp-text">Hyperparameters</span> {# Changed classes #}
          <ChevronDown
            size={20}
            class="ml-auto text-ctp-subtext1 group-open:rotate-180 transition-transform"
          />
        </summary>
        <div class="pt-2 px-4 pb-4 space-y-0"> {# Added px-4 pb-4 for content padding #}
          {#each visibleHyperparameters as param (param.key)}
            <div
              class="flex justify-between items-center py-2.5 px-2 border-b border-ctp-surface0 last:border-b-0 hover:bg-ctp-surface0/50 transition-colors group"
            >
              <!-- Left Column: Key and Info Icon -->
              <div class="flex items-center min-w-0 flex-1 mr-4">
                <span
                  class="text-sm font-medium text-ctp-text truncate mr-1 shrink"
                  title={param.key}>{param.key}</span
                >
                {#if recommendations && recommendations[param.key]}
                  <button
                    class="p-0.5 rounded-sm text-ctp-overlay2 hover:text-ctp-lavender hover:bg-ctp-surface1 transition-colors flex-shrink-0"
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

              <!-- Right Column: Value and Copy Icon -->
              <div class="flex items-center gap-2">
                <span
                  class="text-sm text-ctp-subtext1 truncate max-w-[150px] sm:max-w-xs"
                  title={String(param.value)}>{param.value}</span
                >
                <button
                  class="p-0.5 rounded-sm text-ctp-overlay2 hover:text-ctp-blue hover:bg-ctp-surface1 transition-colors flex-shrink-0"
                  title="Copy value"
                  aria-label="Copy hyperparameter value {param.value}"
                  onclick={() => {
                    navigator.clipboard.writeText(
                      `${param.key}: ${param.value}`,
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
                    <ClipboardCheck size={14} class="text-ctp-green" />
                  {:else}
                    <Copy size={14} />
                  {/if}
                </button>
              </div>
            </div>
          {/each}

          {#if hiddenHyperparameterCount > 0}
            <button
              type="button"
              onclick={showAllHyperparameters}
              class="mt-2 text-xs text-ctp-sky hover:text-ctp-blue hover:underline focus:outline-none w-full text-center hover:bg-ctp-surface0 px-2 py-1 rounded-md transition-colors"
            >
              Show +{hiddenHyperparameterCount} more hyperparameters
            </button>
          {/if}
          {#if allHyperparametersShown && (experiment.hyperparams?.length || 0) > initialHyperparameterLimit}
            <button
              type="button"
              onclick={showLessHyperparameters}
              class="mt-2 text-xs text-ctp-sky hover:text-ctp-blue hover:underline focus:outline-none w-full text-center hover:bg-ctp-surface0 px-2 py-1 rounded-md transition-colors"
            >
              Show less hyperparameters
            </button>
          {/if}
          {#if activeRecommendation}
            <div
              class="mt-3 p-3.5 bg-ctp-surface1 border border-ctp-lavender/50 rounded-lg relative shadow-sm"
            >
              <button
                class="absolute top-2 right-2 p-1 rounded-md text-ctp-subtext1 hover:text-ctp-text hover:bg-ctp-surface2 transition-colors"
                onclick={() => (activeRecommendation = null)}
                aria-label="Close recommendation"
              >
                <X size={15} />
              </button>
              <h4 class="text-sm font-semibold text-ctp-lavender mb-2">
                AI Recommendation
              </h4>
              <p class="text-sm text-ctp-text leading-relaxed">
                {activeRecommendation}
              </p>
            </div>
          {/if}
        </div>
      </details>
    {/if}

    <!-- Metrics section -->
    {#if availableMetrics.length > 0}
      <details class="mt-3 group bg-ctp-mantle/60 border border-ctp-overlay0/30 backdrop-blur-sm rounded-lg" open>
        <summary
          class="flex items-center gap-2.5 cursor-pointer text-ctp-text px-4 py-3 hover:bg-ctp-overlay0/20 transition-colors rounded-t-lg"
        >
          <ChartLine size={18} class="text-ctp-overlay1" />
          <span class="text-lg font-semibold text-ctp-text">Metrics</span> {# Changed classes #}
          <ChevronDown
            size={20}
            class="ml-auto text-ctp-subtext0 group-open:rotate-180"
          />
        </summary>
        <div class="pt-3 px-4 pb-4 space-y-3"> {# Added px-4 pb-4 for content padding #}
          <!-- New Segmented Control -->
          <div class="flex justify-center mb-4">
            <div class="inline-flex bg-ctp-surface0 p-1 rounded-lg space-x-1">
              <button
                class="px-3 py-1.5 rounded-md text-sm font-medium flex items-center gap-2 transition-colors focus:outline-none"
                class:bg-ctp-surface1={!showMetricsTable}
                class:text-ctp-text={!showMetricsTable}
                class:text-ctp-subtext0={showMetricsTable}
                class:hover:text-ctp-text={showMetricsTable}
                class:hover:bg-ctp-surface1={showMetricsTable}
                class:hover:bg-opacity-50={showMetricsTable}
                onclick={() => {
                  showMetricsTable = false;
                }}
              >
                <ChartLine size={16} />
                Chart
              </button>
              <button
                class="px-3 py-1.5 rounded-md text-sm font-medium flex items-center gap-2 transition-colors focus:outline-none"
                class:bg-ctp-surface1={showMetricsTable}
                class:text-ctp-text={showMetricsTable}
                class:text-ctp-subtext0={!showMetricsTable}
                class:hover:text-ctp-text={!showMetricsTable}
                class:hover:bg-ctp-surface1={!showMetricsTable}
                class:hover:bg-opacity-50={!showMetricsTable}
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
                class="flex flex-col justify-center items-center p-6 min-h-[150px] bg-ctp-mantle rounded-lg shadow-sm text-center"
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
              <p
                class="text-sm text-ctp-red bg-ctp-red/10 p-4 rounded-lg shadow-sm border border-ctp-red/30"
              >
                {metricsError}
              </p>
            {:else if rawMetrics.length > 0}
              <div
                class="overflow-x-auto max-h-[500px] border border-ctp-surface1 rounded-lg bg-ctp-mantle shadow-md"
              >
                <table class="w-full text-sm text-left">
                  <thead class="bg-ctp-surface0 sticky top-0 z-10">
                    <tr>
                      <th class="p-3 font-semibold text-ctp-text">Name</th>
                      <th class="p-3 font-semibold text-ctp-text">Value</th>
                      <th class="p-3 font-semibold text-ctp-text">Step</th>
                      <th class="p-3 font-semibold text-ctp-text">Timestamp</th>
                    </tr>
                  </thead>
                  <tbody class="divide-y divide-ctp-surface1">
                    {#each rawMetrics as metric (metric.id)}
                      <tr
                        class="hover:bg-ctp-surface0/70 transition-colors duration-150"
                      >
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
                class="text-sm text-ctp-overlay2 bg-ctp-mantle p-4 rounded-lg shadow-sm text-center"
              >
                No metric data points found for this experiment.
              </p>
            {/if}
          {:else}
            <div class="-mx-4 sm:-mx-6 bg-ctp-mantle p-1 rounded-lg shadow-sm"> {# p-4 to p-1 #}
              <div class="px-2 sm:px-3 w-full overflow-x-auto"> {# This inner px might be redundant if chart handles its own padding well #}
                <InteractiveChart {experiment} />
              </div>
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
