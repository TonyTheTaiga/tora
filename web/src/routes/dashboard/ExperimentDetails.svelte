<script lang="ts">
  import ExperimentChart from "./ExperimentChart.svelte";
  import type { Experiment } from "$lib/types";
  import { copyToClipboard } from "$lib/utils/common";
  import { loading, errors } from "./state.svelte";

  let { experiment } = $props();
  let results = $state<any[]>([]);
  let metricData = $state<Record<string, number[]>>({});

  async function loadExperimentDetails(experiment: Experiment) {
    try {
      loading.experimentDetails = true;
      errors.experimentDetails = null;
      const response = await fetch(`/api/experiments/${experiment.id}/logs`);
      if (!response.ok)
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      const apiResponse = await response.json();
      const metrics = apiResponse.data;
      if (!metrics || !Array.isArray(metrics))
        throw new Error("Invalid response structure from metrics API");

      const resultsByName = new Map<string, any>();
      const seriesByName = new Map<string, any[]>();

      metrics.forEach((m: any) => {
        const mType = m?.metadata?.type ?? "metric";
        if (mType === "result") {
          resultsByName.set(m.name, m);
        } else {
          if (!seriesByName.has(m.name)) seriesByName.set(m.name, []);
          seriesByName.get(m.name)!.push(m);
        }
      });

      const scalarMetricsList: any[] = Array.from(resultsByName.values());
      const computedMetricData: Record<string, number[]> = {};

      seriesByName.forEach((metricList, name) => {
        computedMetricData[name] = metricList
          .sort((a, b) => (a.step || 0) - (b.step || 0))
          .map((m) => m.value);
      });

      results = scalarMetricsList;
      metricData = computedMetricData;
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

<div class="flex flex-col">
  <div
    class="sticky top-0 z-10 surface-elevated border-b border-ctp-surface0/30 p-4"
  >
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

  <div class="p-4">
    {#if loading.experimentDetails}
      <div class="text-center py-12">
        <div class="text-ctp-subtext0 text-sm">
          loading experiment details...
        </div>
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
        {#if results.length > 0}
          <div class="space-y-2">
            <div class="flex items-center gap-2">
              <div class="text-sm text-ctp-text">results</div>
              <div class="text-sm text-ctp-subtext0">[{results.length}]</div>
            </div>
            <div class="border-ctp-terminal-border p-2">
              <div
                class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3"
              >
                {#each results as metric}
                  <div class="flex flex-col gap-1 p-2 hover:bg-ctp-surface0/20">
                    <div
                      class="text-ctp-subtext0 text-[11px] uppercase tracking-wide truncate"
                      title={metric.name}
                    >
                      {metric.name}
                    </div>
                    <div
                      class="text-ctp-text font-semibold tabular-nums font-mono truncate"
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
          </div>
        {/if}

        {#if Object.keys(metricData).length > 0}
          <div class="space-y-2">
            <ExperimentChart {metricData} />
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
            <div class="border-ctp-terminal-border p-3">
              <div
                class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3"
              >
                {#each experiment.hyperparams as param}
                  <div class="flex flex-col gap-1 p-2 hover:bg-ctp-surface0/20">
                    <div
                      class="text-ctp-subtext0 text-[11px] uppercase tracking-wide truncate"
                      title={param.key}
                    >
                      {param.key}
                    </div>
                    <div
                      class="text-ctp-text font-semibold tabular-nums font-mono truncate"
                      title={String(param.value)}
                    >
                      {typeof param.value === "number"
                        ? String(param.value)
                        : String(param.value)}
                    </div>
                  </div>
                {/each}
              </div>
            </div>
          </div>
        {/if}
      </div>
    {/if}
  </div>
</div>
