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
      // Currently showing table, switch to chart view
      showMetricsTable = false;
    } else {
      // Currently showing chart, switch to table view
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
      showMetricsTable = true; // Show the table section (which will handle loader/error/data display internally)
    }
  }
</script>

<article class="h-full bg-ctp-crust rounded-xl shadow-lg flex flex-col overflow-hidden">
  <!-- Header with actions -->
  <header class="px-4 sm:px-6 py-4 bg-ctp-mantle border-b border-ctp-surface1">
    <!-- Combined Header for both Mobile and Desktop -->
    <div class="flex flex-col gap-3">
      <!-- Title and ID Row -->
      <div class="flex items-start justify-between">
        <div class="flex flex-col gap-1 min-w-0 flex-grow"> {/* Changed from items-center to flex-col */}
          <h2
            class="text-xl sm:text-2xl font-semibold text-ctp-text" /* Removed truncate, increased size */
            title={experiment.name}
          >
            {experiment.name}
          </h2>
          <button
            type="button"
            aria-label="Copy Experiment ID"
            title={idCopied ? "ID Copied!" : "Copy Experiment ID"}
            class="flex items-center p-1 rounded-md text-ctp-subtext1 transition-colors hover:bg-ctp-surface0 hover:text-ctp-text active:bg-ctp-surface1 group flex-shrink-0 w-fit" /* Adjusted styling */
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

        <div class="flex items-center gap-2 flex-shrink-0"> {/* Increased gap */}
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
      <div class="flex flex-wrap items-center gap-x-4 gap-y-2 text-ctp-subtext0 text-sm"> {/* Increased font size, gap */}
        <div class="flex items-center gap-1.5"> {/* Increased gap */}
          <Clock size={15} class="flex-shrink-0 text-ctp-overlay1" /> {/* Adjusted icon size and color */}
          <time class="text-ctp-subtext1"> {/* Ensure consistent text color */}
            {new Date(experiment.createdAt)
              .toLocaleString("en-US", { /* Simplified date format */
                year: "numeric",
                month: "short",
                day: "numeric",
                hour: "2-digit",
                minute: "2-digit",
              })}
          </time>
        </div>

        <div
          class="flex items-center gap-1.5 p-1 px-2 rounded-full transition-colors {experiment.visibility === /* Styled as a pill */
          'PUBLIC'
            ? 'text-ctp-green bg-ctp-green/10 hover:bg-ctp-green/20'
            : 'text-ctp-red bg-ctp-red/10 hover:bg-ctp-red/20'}"
          title={experiment.visibility === "PUBLIC" ? "Public" : "Private"}
        >
          {#if experiment.visibility === "PUBLIC"}
            <Globe size={15} /> {/* Adjusted icon size */}
            <span class="text-xs">Public</span>
          {:else}
            <GlobeLock size={15} /> {/* Adjusted icon size */}
            <span class="text-xs">Private</span>
          {/if}
        </div>
      </div>
    </div>
  </header>

  <!-- Content Area -->
  <div class="px-4 sm:px-6 py-4 flex flex-col gap-4 overflow-y-auto flex-grow"> {/* Added more padding and gap, overflow control */}
    <!-- Metadata section -->
    <div
      class="flex flex-col sm:flex-row sm:items-center gap-x-4 gap-y-2 text-ctp-subtext0 text-sm" /* Already updated in previous step, ensure it's correct */
    >
      {#if experiment.tags && experiment.tags.length > 0}
        <div
          class="flex items-center gap-1.5 overflow-x-auto sm:flex-wrap pb-1 sm:pb-0" /* Increased gap */
        >
          <Tag size={15} class="flex-shrink-0 text-ctp-overlay1" /> {/* Adjusted icon and color */}
          <div class="flex gap-1.5 flex-wrap"> {/* Ensured wrap and gap */}
            {#each experiment.tags as tag}
              <span
                class="whitespace-nowrap inline-flex items-center px-2 py-1 text-xs bg-ctp-surface0 text-ctp-blue rounded-full truncate max-w-[150px]" /* Added truncate and max-w */
                title={tag}
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
          description-truncate-detailed
        "
        title={experiment.description}
      >
        {experiment.description}
      </p>
    {/if}
    <!-- Parameters section -->
    {#if experiment.hyperparams && experiment.hyperparams.length > 0}
      <details class="mt-3 group" open> {/* Added mt-3 and open by default */}
        <summary
          class="flex items-center gap-2.5 cursor-pointer text-ctp-text hover:text-ctp-blue py-2.5 rounded-lg -mx-2 px-2 hover:bg-ctp-surface0 transition-colors" /* Enhanced clickable area and feedback */
        >
          <Settings size={18} class="text-ctp-overlay1 flex-shrink-0" /> {/* Consistent icon size */}
          <span class="text-base font-medium">Hyperparameters</span> {/* Clearer heading */}
          <ChevronDown
            size={18}
            class="ml-auto text-ctp-subtext1 group-open:rotate-180 transition-transform" /* Adjusted color and size */
          />
        </summary>
        <div class="pt-3 space-y-3"> {/* Added space-y for better separation */}
          <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {#each experiment.hyperparams as param (param.key)}
              <div
                class="flex items-center bg-ctp-surface0 p-3 rounded-lg overflow-hidden gap-2 shadow-sm" /* Card-like appearance for each param */
              >
                <span
                  class="text-sm font-medium text-ctp-text truncate shrink" /* Main key text */
                  title={param.key}>{param.key}</span
                >
                <span
                  class="ml-auto text-sm text-ctp-subtext1 px-2 py-1 bg-ctp-mantle rounded-md truncate shrink" /* Value styling */
                  title={String(param.value)}>{param.value}</span
                >
                <div class="flex items-center flex-shrink-0 gap-1"> {/* Gap for buttons */}
                  {#if recommendations && recommendations[param.key]}
                    <button
                      class="p-1 rounded-md text-ctp-overlay2 hover:text-ctp-lavender hover:bg-ctp-surface1 transition-colors" /* Consistent button styling */
                      onclick={() => {
                        activeRecommendation =
                          recommendations[param.key].recommendation;
                      }}
                      aria-label="Show recommendation"
                      title="Show AI recommendation"
                    >
                      <Info size={15} /> {/* Consistent icon size */}
                    </button>
                  {/if}
                  <button
                    class="p-1 rounded-md text-ctp-overlay2 hover:text-ctp-blue hover:bg-ctp-surface1 transition-colors" /* Consistent button styling */
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
                      <ClipboardCheck size={15} class="text-ctp-green" /> {/* Consistent icon size */}
                    {:else}
                      <Copy size={15} /> {/* Consistent icon size */}
                    {/if}
                  </button>
                </div>
              </div>
            {/each}
          </div>

          {#if activeRecommendation}
            <div
              class="mt-4 p-3.5 bg-ctp-surface1 border border-ctp-lavender/50 rounded-lg relative shadow-sm" /* Enhanced recommendation box */
            >
              <button
                class="absolute top-2 right-2 p-1 rounded-md text-ctp-subtext1 hover:text-ctp-text hover:bg-ctp-surface2 transition-colors" /* Close button styling */
                onclick={() => (activeRecommendation = null)}
                aria-label="Close recommendation"
              >
                <X size={15} /> {/* Consistent icon size */}
              </button>
              <h4 class="text-sm font-semibold text-ctp-lavender mb-2"> {/* Recommendation title */}
                AI Recommendation
              </h4>
              <p class="text-sm text-ctp-text leading-relaxed"> {/* Recommendation text */}
                {activeRecommendation}
              </p>
            </div>
          {/if}
        </div>
      </details>
    {/if}

    <!-- Metrics section -->
    {#if availableMetrics.length > 0}
      <details class="mt-3 group" open> {/* Added mt-3 and open by default */}
        <summary
          class="flex items-center gap-2.5 cursor-pointer text-ctp-text hover:text-ctp-blue py-2.5 rounded-lg -mx-2 px-2 hover:bg-ctp-surface0 transition-colors" /* Enhanced clickable area and feedback */
        >
          <ChartLine size={18} class="text-ctp-overlay1" /> {/* Consistent icon size */}
          <span class="text-base font-medium">Metrics</span> {/* Clearer heading */}
          <ChevronDown
            size={16}
            class="ml-auto text-ctp-subtext0 group-open:rotate-180"
          />
        </summary>
        <div class="pt-3 space-y-3"> {/* Consistent spacing */}
          <!-- Toggle Button -->
          <div class="mb-4 text-right"> {/* Consistent margin */}
            <button
              class="inline-flex items-center gap-2 text-sm px-3.5 py-2 rounded-lg text-ctp-subtext1 hover:text-ctp-text bg-ctp-surface0 hover:bg-ctp-surface1 transition-colors shadow-sm focus-visible:ring-2 focus-visible:ring-ctp-blue" /* Enhanced button style */
              onclick={toggleMetricsDisplay}
              disabled={metricsLoading && !showMetricsTable}
            >
              {#if showMetricsTable}
                <ChartLine size={16} /> Show Chart {/* Consistent icon size */}
              {:else}
                <Table2 size={16} /> Show Raw Data Table {/* Consistent icon size */}
              {/if}
              {#if metricsLoading && !showMetricsTable}
                <Loader2 size={16} class="animate-spin ml-1" /> {/* Consistent icon size */}
              {/if}
            </button>
          </div>

          <!-- Conditional Display: Chart or Table -->
          {#if showMetricsTable}
            {#if metricsLoading}
              <div class="flex flex-col justify-center items-center p-6 min-h-[150px] bg-ctp-mantle rounded-lg shadow-sm text-center"> {/* Improved loading state look */}
                <Loader2 size={28} class="animate-spin text-ctp-subtext0 mb-3" />
                <span class="text-ctp-subtext0 text-base"
                  >Loading metrics...</span
                >
                <p class="text-ctp-overlay1 text-xs mt-1">Please wait a moment.</p>
              </div>
            {:else if metricsError}
              <p class="text-sm text-ctp-red bg-ctp-red/10 p-4 rounded-lg shadow-sm border border-ctp-red/30"> {/* Enhanced error message style */}
                {metricsError}
              </p>
            {:else if rawMetrics.length > 0}
              <div
                class="overflow-x-auto max-h-[500px] border border-ctp-surface1 rounded-lg bg-ctp-mantle shadow-md" /* Consistent table container style */
              >
                <table class="w-full text-sm text-left">
                  <thead class="bg-ctp-surface0 sticky top-0 z-10"> {/* Table header style */}
                    <tr>
                      <th class="p-3 font-semibold text-ctp-text">Name</th>
                      <th class="p-3 font-semibold text-ctp-text">Value</th>
                      <th class="p-3 font-semibold text-ctp-text">Step</th>
                      <th class="p-3 font-semibold text-ctp-text"
                        >Timestamp</th
                      >
                    </tr>
                  </thead>
                  <tbody class="divide-y divide-ctp-surface1"> {/* Row dividers */}
                    {#each rawMetrics as metric (metric.id)}
                      <tr
                        class="hover:bg-ctp-surface0/70 transition-colors duration-150" /* Row hover effect */
                      >
                        <td
                          class="p-3 text-ctp-subtext1 truncate max-w-sm"
                          title={metric.name}>{metric.name}</td
                        >
                        <td
                          class="p-3 text-ctp-text truncate max-w-sm" /* Added truncate and max-w */
                          title={String(metric.value)}
                          >{typeof metric.value === "number"
                            ? metric.value.toFixed(4)
                            : metric.value}</td
                        >
                        <td class="p-3 text-ctp-subtext1 truncate max-w-[70px]">{metric.step ?? "N/A"}</td> {/* Consistent display for null step, added truncate */}
                        <td class="p-3 text-ctp-subtext1 whitespace-nowrap">
                          {new Date(metric.created_at)
                            .toLocaleString("en-US", { /* Simplified timestamp */
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
              <p class="text-sm text-ctp-overlay2 bg-ctp-mantle p-4 rounded-lg shadow-sm text-center"> {/* Centered message for no data */}
                No metric data points found for this experiment.
              </p>
            {/if}
          {:else}
            <div class="-mx-4 sm:-mx-6 bg-ctp-mantle p-2 rounded-lg shadow-sm"> {/* Chart container with background and padding */}
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

<style>
  .description-truncate-detailed {
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: 5; /* Show 5 lines for detailed view */
    overflow: hidden;
    text-overflow: ellipsis;
  }

  /* Fallback for non-webkit browsers */
  @supports not (-webkit-line-clamp: 5) {
    .description-truncate-detailed {
      max-height: calc(1.5em * 5); /* Assuming line-height ~1.5em, for 5 lines */
      /* white-space: normal; */ /* Ensure it wraps */
    }
  }
</style>
