<script lang="ts">
  import ExperimentChart from "./ExperimentChart.svelte";
  import type { Experiment } from "$lib/types";
  import { copyToClipboard } from "$lib/utils/common";
  import { loading, errors } from "./state.svelte";

  let { experiment } = $props();
  let scalarMetrics = $state<any[]>([]);

  async function loadExperimentDetails(experiment: Experiment) {
    try {
      loading.experimentDetails = true;
      errors.experimentDetails = null;
      const response = await fetch(`/api/experiments/${experiment.id}/metrics`);
      if (!response.ok)
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      const apiResponse = await response.json();
      const metrics = apiResponse.data;
      if (!metrics || !Array.isArray(metrics))
        throw new Error("Invalid response structure from metrics API");

      const metricsByName = new Map<string, any[]>();
      metrics.forEach((metric: any) => {
        if (!metricsByName.has(metric.name)) {
          metricsByName.set(metric.name, []);
        }
        metricsByName.get(metric.name)!.push(metric);
      });

      const scalarMetricsList: any[] = [];
      const timeSeriesNames: string[] = [];
      const metricData: Record<string, number[]> = {};

      metricsByName.forEach((metricList, name) => {
        if (metricList.length === 1) {
          scalarMetricsList.push(metricList[0]);
        } else {
          timeSeriesNames.push(name);
        }

        metricData[name] = metricList
          .sort((a, b) => (a.step || 0) - (b.step || 0))
          .map((m) => m.value);
      });

      experiment.availableMetrics = timeSeriesNames;
      experiment.metricData = metricData;
      scalarMetrics = scalarMetricsList;
    } catch (error) {
      errors.experimentDetails =
        error instanceof Error
          ? error.message
          : "Failed to load experiment details";
    } finally {
      loading.experimentDetails = false;
    }
  }

  $effect(() => {
    loadExperimentDetails(experiment);
  });
</script>

<div class="terminal-chrome-header">
  <div class="mb-3">
    <h2 class="text-ctp-text font-medium text-lg mb-2">
      {experiment.name}
    </h2>
    {#if experiment.description}
      <p class="text-ctp-subtext0 line-clamp-2 mb-3 text-sm">
        {experiment.description}
      </p>
    {/if}
    <button
      class="text-xs text-ctp-overlay0 hover:text-ctp-blue"
      onclick={() => copyToClipboard(experiment.id)}
      title="click to copy experiment id"
    >
      experiment id: {experiment.id}
    </button>
  </div>
</div>

<div class="overflow-y-auto">
  {#if loading.experimentDetails}
    <div class="text-center py-12">
      <div class="text-ctp-subtext0 text-sm">loading experiment details...</div>
    </div>
  {:else if errors.experimentDetails}
    <div class="surface-layer-2 p-4">
      <div class="text-ctp-red font-medium text-sm mb-3">
        error loading experiment details
      </div>
      <div class="text-ctp-subtext0 mb-4 text-xs">
        {errors.experimentDetails}
      </div>
      >
    </div>
  {:else}
    <div class="space-y-6">
      {#if experiment.availableMetrics.length > 0}
        <div class="space-y-2">
          <div class="bg-ctp-surface0/10 border border-ctp-surface0/20 p-4">
            <ExperimentChart
              metricData={experiment.metricData}
              availableMetrics={experiment.availableMetrics}
            />
          </div>
        </div>
      {/if}

      {#if scalarMetrics.length > 0}
        <div class="space-y-2">
          <div class="flex items-center gap-2">
            <div class="text-sm text-ctp-text">scalar metrics</div>
            <div class="text-sm text-ctp-subtext0">
              [{scalarMetrics.length}]
            </div>
          </div>
          <div class="terminal-chrome">
            {#each scalarMetrics as metric, index}
              <div
                class="flex text-sm hover:bg-ctp-surface0/20 p-3 {index !==
                experiment.availableMetrics.length - 1
                  ? 'border-b border-ctp-surface0/20'
                  : ''} {index % 2 === 0 ? 'bg-ctp-surface0/5' : ''}"
              >
                <div class="w-4 text-ctp-green">•</div>
                <div class="flex-1 text-ctp-text truncate" title={metric.name}>
                  {metric.name}
                </div>
                <div
                  class="w-24 text-right text-ctp-blue"
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
      {/if}

      {#if experiment.tags?.length}
        <div class="space-y-2">
          <div class="flex items-center gap-2">
            <div class="text-sm text-ctp-text">tags</div>
            <div class="text-sm text-ctp-subtext0">
              [{experiment.tags.length}]
            </div>
          </div>
          <div class="flex flex-wrap gap-1">
            {#each experiment.tags as tag}
              <span
                class="text-xs bg-ctp-blue/20 text-ctp-blue border border-ctp-blue/30 px-2 py-1"
                >{tag}</span
              >
            {/each}
          </div>
        </div>
      {/if}

      {#if experiment.hyperparams?.length}
        <div class="space-y-2">
          <div class="flex items-center gap-2">
            <div class="text-sm text-ctp-text">hyperparameters</div>
            <div class="text-sm text-ctp-subtext0">
              [{experiment.hyperparams.length}]
            </div>
          </div>
          <div class="terminal-chrome">
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-0">
              {#each experiment.hyperparams as param, index}
                <div
                  class="flex flex-row items-center justify-between hover:bg-ctp-surface0/20 px-3 py-2 text-sm gap-2 {index !==
                  experiment.hyperparams.length - 1
                    ? 'border-b border-ctp-surface0/20'
                    : ''} {index % 2 === 0 ? 'bg-ctp-surface0/5' : ''}"
                >
                  <span class="text-ctp-subtext0 truncate">{param.key}</span>
                  <span
                    class="text-ctp-blue bg-ctp-surface0/20 border border-ctp-surface0/30 px-2 py-1 max-w-32 truncate text-xs"
                    title={String(param.value)}>{param.value}</span
                  >
                </div>
              {/each}
            </div>
          </div>
        </div>
      {/if}
    </div>
  {/if}
</div>
