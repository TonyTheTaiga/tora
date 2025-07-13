<script lang="ts">
  import InteractiveChart from "./interactive-chart.svelte";
  import type { Experiment } from "$lib/types";

  let {
    experiment,
    scalarMetrics,
    timeSeriesNames,
    initialLimit = 10,
  }: {
    experiment: Experiment;
    scalarMetrics: Array<{ name: string; value: any }>;
    timeSeriesNames: string[];
    initialLimit?: number;
  } = $props();

  let experimentWithMetrics = $derived({
    ...experiment,
    availableMetrics: timeSeriesNames,
  });

  let showAllScalarMetrics = $state(false);
  let metricsView = $state<"chart" | "data">("chart");

  let visibleScalarMetrics = $derived(
    showAllScalarMetrics || scalarMetrics.length <= initialLimit
      ? scalarMetrics
      : scalarMetrics.slice(0, initialLimit),
  );
</script>

{#if scalarMetrics.length > 0 || timeSeriesNames.length > 0}
  <div class="space-y-2">
    <div class="flex items-center gap-2">
      <div class="text-sm text-ctp-text">metrics</div>
      <div class="text-sm text-ctp-subtext0 font-mono">
        [{scalarMetrics.length + timeSeriesNames.length}]
      </div>
      <div class="flex items-center gap-1">
        <button
          onclick={() => (metricsView = "chart")}
          class="text-sm text-ctp-{metricsView === 'chart'
            ? 'blue'
            : 'subtext0'} hover:text-ctp-blue transition-colors"
        >
          [chart]
        </button>
        <button
          onclick={() => (metricsView = "data")}
          class="text-sm text-ctp-{metricsView === 'data'
            ? 'blue'
            : 'subtext0'} hover:text-ctp-blue transition-colors"
        >
          [table]
        </button>
        <a
          href={`/api/experiments/${experiment.id}/metrics/csv`}
          class="text-sm text-ctp-subtext0 hover:text-ctp-blue transition-colors"
          download
        >
          [csv]
        </a>
      </div>
    </div>
  </div>
{/if}

{#if metricsView === "chart" && timeSeriesNames.length > 0}
  <div class="space-y-2">
    <div class="bg-ctp-surface0/10 border border-ctp-surface0/20 p-2 md:p-4">
      <InteractiveChart experiment={experimentWithMetrics} />
    </div>
  </div>
{/if}

{#if metricsView === "data" && scalarMetrics.length > 0}
  <div class="space-y-2">
    <div class="flex items-center gap-2">
      <div class="text-sm text-ctp-text">scalar metrics</div>
      <div class="text-sm text-ctp-subtext0 font-mono">
        [{scalarMetrics.length}]
      </div>
    </div>
    <div class="bg-ctp-surface0/10 border border-ctp-surface0/20">
      <div class="hidden md:block">
        <div
          class="flex text-sm text-ctp-subtext0 p-3 border-b border-ctp-surface0/20 sticky top-0"
        >
          <div class="w-4"></div>
          <div class="flex-1">metric</div>
          <div class="w-20 text-right">value</div>
        </div>

        <div class="{showAllScalarMetrics ? '' : 'max-h-60'} overflow-y-auto">
          {#each visibleScalarMetrics as metric}
            <div
              class="flex text-sm hover:bg-ctp-surface0/20 p-3 transition-colors border-b border-ctp-surface0/5"
            >
              <div class="w-4 text-ctp-green"></div>
              <div class="flex-1 text-ctp-text truncate" title={metric.name}>
                {metric.name}
              </div>
              <div
                class="w-20 text-right text-ctp-blue font-mono"
                title={String(metric.value)}
              >
                {typeof metric.value === "number"
                  ? metric.value.toFixed(4)
                  : metric.value}
              </div>
            </div>
          {/each}
        </div>
      </div>
      <div
        class="md:hidden {showAllScalarMetrics
          ? ''
          : 'max-h-60'} overflow-y-auto"
      >
        {#each visibleScalarMetrics as metric}
          <div
            class="border-b border-ctp-surface0/5 p-3 hover:bg-ctp-surface0/20 transition-colors"
          >
            <div class="flex items-center justify-between mb-1">
              <div class="flex items-center gap-2">
                <div class="text-ctp-green text-sm"></div>
                <div
                  class="text-sm text-ctp-text font-mono truncate"
                  title={metric.name}
                >
                  {metric.name}
                </div>
              </div>
            </div>
            <div class="flex items-center justify-between text-sm">
              <div class="text-ctp-blue font-mono" title={String(metric.value)}>
                {typeof metric.value === "number"
                  ? metric.value.toFixed(4)
                  : metric.value}
              </div>
            </div>
          </div>
        {/each}
      </div>

      {#if scalarMetrics.length > initialLimit}
        <button
          onclick={() => (showAllScalarMetrics = !showAllScalarMetrics)}
          class="w-full text-sm text-ctp-subtext0 hover:text-ctp-text p-3 text-center border-t border-ctp-surface0/20 transition-colors"
        >
          {showAllScalarMetrics
            ? "show less"
            : `show ${scalarMetrics.length - initialLimit} more`}
        </button>
      {/if}
    </div>
  </div>
{/if}
