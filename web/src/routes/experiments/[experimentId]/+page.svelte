<script lang="ts">
  import type { PageData } from "./$types";
  import ExperimentHeader from "./experiment-header.svelte";
  import ExperimentTags from "./experiment-tags.svelte";
  import ExperimentHyperparams from "./experiment-hyperparams.svelte";
  import ExperimentMetrics from "./experiment-metrics.svelte";
  import ExperimentSystemInfo from "./experiment-system-info.svelte";

  let { data }: { data: PageData } = $props();
  let { experiment, scalarMetrics, timeSeriesNames } = $derived(data);

  const initialLimit = 10;
</script>

<div class="font-mono">
  <ExperimentHeader {experiment} />

  <div class="p-4 md:p-6 space-y-4 md:space-y-6">
    <ExperimentMetrics
      {experiment}
      {scalarMetrics}
      {timeSeriesNames}
      {initialLimit}
    />

    <ExperimentTags tags={experiment.tags || []} {initialLimit} />

    <ExperimentHyperparams
      hyperparams={experiment.hyperparams || []}
      {initialLimit}
    />

    <ExperimentSystemInfo {experiment} />
  </div>
</div>
