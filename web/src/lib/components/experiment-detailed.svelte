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

  let availableMetrics = $derived(
    experiment.metricData
      ? Object.keys(experiment.metricData)
      : experiment.availableMetrics || [],
  );

  async function toggleMetricsDisplay() {
    if (showMetricsTable) {
      showMetricsTable = false;
    } else {
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
          rawMetrics = data as Metric[];
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
          rawMetrics = [];
        } finally {
          metricsLoading = false;
        }
      }
      showMetricsTable = true;
    }
  }
</script>

<article
  class="h-full bg-ctp-crust rounded-xl shadow-lg flex flex-col overflow-hidden"
>
  <!-- Header with actions -->
  <header class="px-4 sm:px-6 py-4 bg-ctp-mantle border-b border-ctp-surface1">
    <!-- Combined Header for both Mobile and Desktop -->
    <div class="flex flex-col gap-3">
      <!-- Title and ID Row -->
      <div class="flex items-start justify-between">
        <div class="flex flex-col gap-1 min-w-0 flex-grow">
          <h2
            class="text-xl sm:text-2xl font-semibold text-ctp-text"
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
  <div class="px-4 sm:px-6 py-4 flex flex-col gap-4 overflow-y-auto flex-grow">
    <!-- Metadata section -->
    <div
      class="flex flex-col sm:flex-row sm:items-center gap-x-4 gap-y-2 text-ctp-subtext0 text-sm"
    >
      {#if experiment.tags && experiment.tags.length > 0}
        <div
          class="flex items-center gap-1.5 overflow-x-auto sm:flex-wrap pb-1 sm:pb-0"
        >
          <Tag size={15} class="flex-shrink-0 text-ctp-overlay1" />
          <div class="flex gap-1.5 flex-wrap">
            {#each experiment.tags as tag}
              <span
                class="whitespace-nowrap inline-flex items-center px-2 py-1 text-xs bg-ctp-surface0 text-ctp-blue rounded-full"
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
      <details class="mt-3 group" open>
        <summary
          class="flex items-center gap-2.5 cursor-pointer text-ctp-text hover:text-ctp-blue py-2.5 rounded-lg -mx-2 px-2 hover:bg-ctp-surface0 transition-colors"
        >
          <Settings size={18} class="text-ctp-overlay1 flex-shrink-0" />
          <span class="text-base font-medium">Hyperparameters</span>
          <ChevronDown
            size={18}
            class="ml-auto text-ctp-subtext1 group-open:rotate-180 transition-transform"
          />
        </summary>
        <div class="pt-3 space-y-3">
          <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {#each experiment.hyperparams as param (param.key)}
              <div
                class="flex items-center bg-ctp-surface0 p-3 rounded-lg overflow-hidden gap-2 shadow-sm"
              >
                <span
                  class="text-sm font-medium text-ctp-text truncate shrink"
                  title={param.key}>{param.key}</span
                >
                <span
                  class="ml-auto text-sm text-ctp-subtext1 px-2 py-1 bg-ctp-mantle rounded-md truncate shrink"
                  title={String(param.value)}>{param.value}</span
                >
                <div class="flex items-center flex-shrink-0 gap-1">
                  {#if recommendations && recommendations[param.key]}
                    <button
                      class="p-1 rounded-md text-ctp-overlay2 hover:text-ctp-lavender hover:bg-ctp-surface1 transition-colors"
                      onclick={() => {
                        activeRecommendation =
                          recommendations[param.key].recommendation;
                      }}
                      aria-label="Show recommendation"
                      title="Show AI recommendation"
                    >
                      <Info size={15} />
                    </button>
                  {/if}
                  <button
                    class="p-1 rounded-md text-ctp-overlay2 hover:text-ctp-blue hover:bg-ctp-surface1 transition-colors"
                    title="Copy {param.key}: {param.value}"
                    aria-label="Copy hyperparameter {param.key}"
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
                      <ClipboardCheck size={15} class="text-ctp-green" />
                    {:else}
                      <Copy size={15} />
                    {/if}
                  </button>
                </div>
              </div>
            {/each}
          </div>

          {#if activeRecommendation}
            <div
              class="mt-4 p-3.5 bg-ctp-surface1 border border-ctp-lavender/50 rounded-lg relative shadow-sm"
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
      <details class="mt-3 group" open>
        <summary
          class="flex items-center gap-2.5 cursor-pointer text-ctp-text hover:text-ctp-blue py-2.5 rounded-lg -mx-2 px-2 hover:bg-ctp-surface0 transition-colors"
        >
          <ChartLine size={18} class="text-ctp-overlay1" />
          <span class="text-base font-medium">Metrics</span>
          <ChevronDown
            size={16}
            class="ml-auto text-ctp-subtext0 group-open:rotate-180"
          />
        </summary>
        <div class="pt-3 space-y-3">
          <!-- Toggle Button -->
          <div class="mb-4 text-right">
            <button
              class="inline-flex items-center gap-2 text-sm px-3.5 py-2 rounded-lg text-ctp-subtext1 hover:text-ctp-text bg-ctp-surface0 hover:bg-ctp-surface1 transition-colors shadow-sm focus-visible:ring-2 focus-visible:ring-ctp-blue"
              onclick={toggleMetricsDisplay}
              disabled={metricsLoading && !showMetricsTable}
            >
              {#if showMetricsTable}
                <ChartLine size={16} /> Show Chart
              {:else}
                <Table2 size={16} /> Show Raw Data Table
              {/if}
              {#if metricsLoading && !showMetricsTable}
                <Loader2 size={16} class="animate-spin ml-1" />
              {/if}
            </button>
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
                          class="p-3 text-ctp-text"
                          title={String(metric.value)}
                          >{typeof metric.value === "number"
                            ? metric.value.toFixed(4)
                            : metric.value}</td
                        >
                        <td class="p-3 text-ctp-subtext1"
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
            <div class="-mx-4 sm:-mx-6 bg-ctp-mantle p-2 rounded-lg shadow-sm">
              <div class="px-2 sm:px-3 w-full overflow-x-auto">
                <InteractiveChart {experiment} />
              </div>
            </div>
          {/if}
        </div>
      </details>
    {/if}
  </div>
</article>
