<script lang="ts">
  import type { PageData } from "./$types";
  import ExperimentHeader from "./experiment-header.svelte";
  import ExperimentTags from "./experiment-tags.svelte";
  import ExperimentHyperparams from "./experiment-hyperparams.svelte";
  import ExperimentMetrics from "./experiment-metrics.svelte";
  import ExperimentSystemInfo from "./experiment-system-info.svelte";

  let { data }: { data: PageData } = $props();
  let { experiment, scalarMetrics, timeSeriesNames } = $derived(data);

  let copiedId = $state(false);
  let copiedMetric = $state<string | null>(null);
  let copiedParam = $state<string | null>(null);

  const initialLimit = 10;

  function copyToClipboard(
    text: string,
    type: "id" | "metric" | "param",
    key?: string,
  ) {
    navigator.clipboard.writeText(text);
    if (type === "id") {
      copiedId = true;
      setTimeout(() => (copiedId = false), 1200);
    } else if (type === "metric" && key) {
      copiedMetric = key;
      setTimeout(() => (copiedMetric = null), 1200);
    } else if (type === "param" && key) {
      copiedParam = key;
      setTimeout(() => (copiedParam = null), 1200);
    }
  }

  function handleCopyId(id: string) {
    copyToClipboard(id, "id");
  }

  function handleCopyMetric(value: string, key: string) {
    copyToClipboard(value, "metric", key);
  }

  function handleCopyParam(value: string, key: string) {
    copyToClipboard(value, "param", key);
  }
</script>

<div class="font-mono">
  <ExperimentHeader {experiment} onCopyId={handleCopyId} />

  <div class="p-4 md:p-6 space-y-4 md:space-y-6">
    <ExperimentMetrics
      {experiment}
      {scalarMetrics}
      {timeSeriesNames}
      {initialLimit}
      onCopyMetric={handleCopyMetric}
      {copiedMetric}
    />

    <ExperimentTags tags={experiment.tags || []} {initialLimit} />

    <ExperimentHyperparams
      hyperparams={experiment.hyperparams || []}
      {initialLimit}
      onCopyParam={handleCopyParam}
      {copiedParam}
    />

    <ExperimentSystemInfo {experiment} />
  </div>
</div>
