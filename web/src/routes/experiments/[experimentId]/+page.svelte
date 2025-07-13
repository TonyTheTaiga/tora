<script lang="ts">
  import type { PageData } from "./$types";
  import { PageHeader } from "$lib/components";
  import ExperimentTags from "./experiment-tags.svelte";
  import ExperimentHyperparams from "./experiment-hyperparams.svelte";
  import ExperimentMetrics from "./experiment-metrics.svelte";
  import ExperimentSystemInfo from "./experiment-system-info.svelte";

  let { data }: { data: PageData } = $props();
  let { experiment, scalarMetrics, timeSeriesNames } = $derived(data);

  const initialLimit = 10;

  function copyToClipboard(text: string) {
    navigator.clipboard.writeText(text);
  }
</script>

<div class="font-mono">
  <PageHeader
    title={experiment.name}
    description={experiment.description || undefined}
    additionalInfo={experiment.id}
    onAdditionalInfoClick={() => copyToClipboard(experiment.id)}
    additionalInfoTitle="click to copy experiment id"
  />

  <div class="p-2 sm:p-4 md:p-6 space-y-3 sm:space-y-4 md:space-y-6">
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
