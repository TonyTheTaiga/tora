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
    Pencil,
    Info,
    Eye,
    EyeClosed,
    Sparkle,
    ClipboardCheck,
    Copy,
    Loader2,
    Globe,
    GlobeLock,
  } from "lucide-svelte";
  import InteractiveChart from "./interactive-chart.svelte";
  import {
    openDeleteExperimentModal,
    openEditExperimentModal,
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
        rawMetrics = data as Metric[];
        if (rawMetrics.length === 0) {
          metricsError = "No raw metric data points found for this experiment.";
        }
      } catch (err) {
        if (err instanceof Error) {
          metricsError = err.message;
        } else {
          metricsError = "An unknown error occurred while fetching metrics.";
        }
        rawMetrics = [];
      } finally {
        metricsLoading = false;
      }
    }
  }
</script>

<!-- Folder tab with actions -->
<div class="flex items-center justify-end mb-2 font-mono">
  <div
    class="flex items-center gap-1 bg-ctp-surface0/20 border border-ctp-surface0/30 px-2 py-1"
  >
    <button
      onclick={async () => {
        const response = await fetch(
          `/api/ai/analysis?experimentId=${experiment.id}`,
        );
        const data = (await response.json()) as ExperimentAnalysis;
        recommendations = data.hyperparameter_recommendations;
      }}
      class="text-ctp-subtext0 hover:text-ctp-lavender hover:bg-ctp-surface0/30 p-1 transition-all"
      title="Get AI recommendations"
    >
      <Sparkle size={14} />
    </button>
    <button
      onclick={() => {
        openEditExperimentModal(experiment);
      }}
      class="text-ctp-subtext0 hover:text-ctp-blue hover:bg-ctp-surface0/30 p-1 transition-all"
      title="Edit experiment"
    >
      <Pencil size={14} />
    </button>
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
      class="text-ctp-subtext0 hover:text-ctp-teal hover:bg-ctp-surface0/30 p-1 transition-all"
      title="Show experiment chain"
    >
      {#if highlighted.includes(experiment.id)}
        <EyeClosed size={14} />
      {:else}
        <Eye size={14} />
      {/if}
    </button>
    <button
      onclick={(e) => {
        e.stopPropagation();
        openDeleteExperimentModal(experiment);
      }}
      class="text-ctp-subtext0 hover:text-ctp-red hover:bg-ctp-surface0/30 p-1 transition-all"
      title="Delete experiment"
    >
      <X size={14} />
    </button>
    <div
      class="p-1 text-ctp-subtext0"
      title={experiment.visibility === "PUBLIC" ? "Public" : "Private"}
    >
      {#if experiment.visibility === "PUBLIC"}
        <Globe size={14} class="text-ctp-green" />
      {:else}
        <GlobeLock size={14} class="text-ctp-red" />
      {/if}
    </div>
  </div>
</div>

<!-- Terminal-style experiment details -->
<div class="font-mono space-y-3">
  <!-- Header section - file listing style -->
  <div class="space-y-3">
    <!-- Primary info - name as filename -->
    <div class="flex items-center gap-2">
      <div class="text-ctp-green text-sm">●</div>
      <div class="text-sm text-ctp-text font-mono font-semibold break-words min-w-0">
        {experiment.name}
      </div>
      <div class="text-xs text-ctp-subtext0 font-mono ml-auto">
        {new Date(experiment.createdAt).toLocaleDateString("en-US", {
          month: "short",
          day: "numeric",
          year: "2-digit",
        })}
      </div>
    </div>

    <!-- Secondary metadata -->
    <div class="pl-6 space-y-1 text-xs font-mono">
      <div class="flex items-center gap-2">
        <span class="text-ctp-subtext0 w-8">id:</span>
        <button
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
          class="text-ctp-blue hover:text-ctp-blue/80 transition-colors flex items-center gap-1 min-w-0"
          title={idCopied ? "ID Copied!" : "Copy Experiment ID"}
        >
          <span class="truncate">{experiment.id}</span>
          {#if idCopied}
            <ClipboardCheck size={10} class="text-ctp-green flex-shrink-0" />
          {:else}
            <Copy size={10} class="flex-shrink-0" />
          {/if}
        </button>
      </div>
      
      {#if experiment.description}
        <div class="flex gap-2">
          <span class="text-ctp-subtext0 w-8 flex-shrink-0">desc:</span>
          <span class="text-ctp-subtext1 break-words min-w-0">
            {experiment.description}
          </span>
        </div>
      {/if}
    </div>
  </div>

  <!-- Tags section -->
  {#if experiment.tags && experiment.tags.length > 0}
    <div class="space-y-1">
      <div class="flex flex-wrap gap-1">
        {#each visibleTags as tag}
          <span
            class="text-xs bg-ctp-blue/20 text-ctp-blue border border-ctp-blue/30 px-2 py-0.5 rounded-full font-mono"
          >
            {tag}
          </span>
        {/each}
        {#if hiddenTagCount > 0}
          <button
            onclick={showAllTags}
            class="text-xs text-ctp-subtext0 hover:text-ctp-blue transition-colors px-1 py-0.5"
          >
            +{hiddenTagCount}
          </button>
        {/if}
        {#if allTagsShown && experiment.tags.length > initialTagLimit}
          <button
            onclick={showLessTags}
            class="text-xs text-ctp-subtext0 hover:text-ctp-blue transition-colors px-1 py-0.5"
          >
            less
          </button>
        {/if}
      </div>
    </div>
  {/if}

  <!-- Hyperparameters section -->
  {#if experiment.hyperparams && experiment.hyperparams.length > 0}
    <div class="space-y-1">
      <div class="flex items-center justify-between">
        <div class="text-sm text-ctp-text font-medium">hyperparams</div>
        <div class="text-xs text-ctp-subtext0 font-mono">
          [{experiment.hyperparams?.length || 0}]
        </div>
      </div>
      <div class="space-y-1">
        {#each visibleHyperparameters as param (param.key)}
          <div
            class="flex items-center justify-between hover:bg-ctp-surface0/20 px-1 py-1 transition-colors text-xs min-w-0"
          >
            <div class="flex items-center gap-1 flex-1 min-w-0">
              <span class="text-ctp-subtext0 truncate">{param.key}</span>
              <span class="text-ctp-text">=</span>
              {#if recommendations && recommendations[param.key]}
                <button
                  onclick={() => {
                    activeRecommendation =
                      recommendations[param.key].recommendation;
                  }}
                  class="text-ctp-lavender hover:text-ctp-lavender/80 transition-colors flex-shrink-0"
                  title="Show AI recommendation"
                >
                  <Info size={10} />
                </button>
              {/if}
            </div>
            <div class="flex items-center gap-1 flex-shrink-0">
              <span
                class="text-ctp-blue font-mono bg-ctp-surface0/20 border border-ctp-surface0/30 px-1 py-0.5 text-xs max-w-24 truncate"
                title={String(param.value)}
              >
                {param.value}
              </span>
              <button
                onclick={() => {
                  navigator.clipboard.writeText(String(param.value));
                  copiedParamKey = param.key;
                  setTimeout(() => {
                    if (copiedParamKey === param.key) {
                      copiedParamKey = null;
                    }
                  }, 1200);
                }}
                class="text-ctp-subtext0 hover:text-ctp-text transition-colors"
                title="Copy value"
              >
                {#if copiedParamKey === param.key}
                  <ClipboardCheck size={10} class="text-ctp-green" />
                {:else}
                  <Copy size={10} />
                {/if}
              </button>
            </div>
          </div>
        {/each}

        {#if (experiment.hyperparams?.length || 0) > initialHyperparameterLimit}
          <button
            onclick={() => {
              allHyperparametersShown = !allHyperparametersShown;
            }}
            class="w-full text-xs text-ctp-subtext0 hover:text-ctp-text px-1 py-1 text-center border-t border-ctp-surface0/20 transition-colors"
          >
            {allHyperparametersShown ? "less" : `+${hiddenHyperparameterCount}`}
          </button>
        {/if}

        {#if activeRecommendation}
          <div
            class="mt-2 p-2 bg-ctp-lavender/10 border border-ctp-lavender/30 relative"
          >
            <button
              onclick={() => (activeRecommendation = null)}
              class="absolute top-1 right-1 text-ctp-subtext0 hover:text-ctp-text transition-colors"
            >
              <X size={10} />
            </button>
            <div class="text-xs text-ctp-lavender mb-1 font-mono">ai_recommendation:</div>
            <div class="text-xs text-ctp-text leading-relaxed pr-4 font-mono">
              {activeRecommendation}
            </div>
          </div>
        {/if}
      </div>
    </div>
  {/if}

  <!-- Metrics section -->
  {#if availableMetrics.length > 0}
    <div class="space-y-1">
      <div class="flex items-center gap-2">
        <div class="text-sm text-ctp-text font-medium">metrics</div>
        <div class="flex items-center gap-1 text-xs font-mono">
          <button
            onclick={() => {
              showMetricsTable = false;
            }}
            class="transition-colors {!showMetricsTable
              ? 'text-ctp-blue'
              : 'text-ctp-subtext0 hover:text-ctp-text'}"
          >
            [chart]
          </button>
          <button
            onclick={() => {
              showMetricsTable = true;
              fetchRawMetricsIfNeeded();
            }}
            class="transition-colors {showMetricsTable
              ? 'text-ctp-blue'
              : 'text-ctp-subtext0 hover:text-ctp-text'}"
            disabled={metricsLoading}
          >
            [data]
          </button>
        </div>
      </div>

      {#if showMetricsTable}
        {#if metricsLoading}
          <div class="flex items-center justify-center p-4 text-center">
            <Loader2 size={16} class="animate-spin text-ctp-subtext0 mr-2" />
            <span class="text-ctp-subtext0 text-xs">loading...</span>
          </div>
        {:else if metricsError}
          <div class="text-xs text-ctp-red p-2">
            error: {metricsError}
          </div>
        {:else if rawMetrics.length > 0}
          <div class="overflow-x-auto max-h-80 font-mono">
            <!-- Terminal-style table header -->
            <div
              class="flex text-xs text-ctp-subtext0 pb-1 border-b border-ctp-surface0/20 sticky top-0"
            >
              <div class="w-3">•</div>
              <div class="flex-1 min-w-0">name</div>
              <div class="w-16 text-right">value</div>
              <div class="w-12 text-right">step</div>
              <div class="w-16 text-right hidden sm:block">time</div>
            </div>

            <!-- Metrics entries -->
            {#each rawMetrics as metric (metric.id)}
              <div
                class="flex text-xs hover:bg-ctp-surface0/20 py-0.5 transition-colors min-w-0"
              >
                <div class="w-3 text-ctp-green">●</div>
                <div
                  class="flex-1 text-ctp-text truncate min-w-0"
                  title={metric.name}
                >
                  {metric.name}
                </div>
                <div
                  class="w-16 text-right text-ctp-blue font-mono truncate"
                  title={String(metric.value)}
                >
                  {typeof metric.value === "number"
                    ? metric.value.toFixed(2)
                    : metric.value}
                </div>
                <div class="w-12 text-right text-ctp-subtext0">
                  {metric.step ?? "N/A"}
                </div>
                <div class="w-16 text-right text-ctp-subtext0 hidden sm:block">
                  {new Date(metric.created_at).toLocaleDateString("en-US", {
                    month: "numeric",
                    day: "numeric",
                  })}
                </div>
              </div>
            {/each}
          </div>
        {:else}
          <div class="text-xs text-ctp-subtext0 text-center p-2">
            no data found
          </div>
        {/if}
      {:else}
        <InteractiveChart {experiment} />
      {/if}
    </div>
  {/if}
</div>
