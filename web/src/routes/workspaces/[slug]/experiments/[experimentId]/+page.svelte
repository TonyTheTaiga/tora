<script lang="ts">
  import { page } from "$app/state";
  import {
    Copy,
    ClipboardCheck,
    Globe,
    GlobeLock,
    FileText,
    Database,
    Activity,
    Settings,
    Clock,
    Tag,
    Hash,
    User,
    Calendar,
    Eye,
    Download,
    ExternalLink,
  } from "lucide-svelte";
  import type { PageData } from "./$types";
  import InteractiveChart from "../../interactive-chart.svelte";

  let { data }: { data: PageData } = $props();
  let {
    experiment,
    scalarMetrics,
    timeSeriesMetrics,
    timeSeriesNames,
    files,
    workspace,
  } = $derived(data);

  // Create enhanced experiment object with time series metrics for the chart
  let experimentWithMetrics = $derived({
    ...experiment,
    availableMetrics: timeSeriesNames,
  });

  let copiedId = $state(false);
  let copiedMetric = $state<string | null>(null);
  let copiedParam = $state<string | null>(null);
  let showAllScalarMetrics = $state(false);
  let showAllTimeSeriesMetrics = $state(false);
  let showAllParams = $state(false);
  let showAllTags = $state(false);
  let metricsView = $state<"chart" | "data">("chart");

  const initialLimit = 10;

  let visibleScalarMetrics = $derived(
    showAllScalarMetrics || scalarMetrics.length <= initialLimit
      ? scalarMetrics
      : scalarMetrics.slice(0, initialLimit),
  );

  let visibleTimeSeriesMetrics = $derived(
    showAllTimeSeriesMetrics || timeSeriesMetrics.length <= initialLimit * 2
      ? timeSeriesMetrics
      : timeSeriesMetrics.slice(0, initialLimit * 2),
  );

  let visibleParams = $derived(
    showAllParams || (experiment.hyperparams?.length || 0) <= initialLimit
      ? experiment.hyperparams || []
      : (experiment.hyperparams || []).slice(0, initialLimit),
  );

  let visibleTags = $derived(
    showAllTags || (experiment.tags?.length || 0) <= initialLimit
      ? experiment.tags || []
      : (experiment.tags || []).slice(0, initialLimit),
  );

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

  function formatFileSize(bytes: number): string {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
  }

  function getFileIcon(filename: string) {
    const ext = filename.split(".").pop()?.toLowerCase();
    switch (ext) {
      case "json":
      case "yml":
      case "yaml":
      case "toml":
      case "ini":
        return Settings;
      case "csv":
      case "tsv":
      case "parquet":
        return Database;
      case "png":
      case "jpg":
      case "jpeg":
      case "gif":
      case "svg":
        return Eye;
      case "py":
      case "js":
      case "ts":
      case "go":
      case "rs":
        return FileText;
      default:
        return FileText;
    }
  }
</script>

<div class="bg-ctp-base font-mono">
  <div class="p-4 md:p-6 space-y-4 md:space-y-6">
    <!-- Primary experiment info -->
    <div class="space-y-3">
      <!-- Header section - file listing style -->
      <div class="flex flex-col sm:flex-row sm:items-center gap-2">
        <div class="flex items-center gap-2 min-w-0 flex-1">
          <div class="text-ctp-green text-sm">●</div>
          <div class="text-base md:text-lg text-ctp-text break-words min-w-0">
            {experiment.name}
          </div>
        </div>
        <div class="flex items-center gap-2 ml-6 sm:ml-0">
          <div class="text-xs text-ctp-subtext0">
            {new Date(experiment.createdAt).toLocaleDateString("en-US", {
              month: "short",
              day: "numeric",
              year: "2-digit",
              hour: "2-digit",
              minute: "2-digit",
            })}
          </div>
          {#if experiment.visibility === "PUBLIC"}
            <Globe size={12} class="text-ctp-green md:w-4 md:h-4" />
          {:else}
            <GlobeLock size={12} class="text-ctp-red md:w-4 md:h-4" />
          {/if}
        </div>
      </div>

      <!-- Metadata grid -->
      <div
        class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 md:gap-4 pl-4 md:pl-6"
      >
        <!-- ID -->
        <div class="space-y-1">
          <div class="text-xs text-ctp-subtext0 flex items-center gap-1">
            <Hash size={10} />
            id
          </div>
          <button
            onclick={() => copyToClipboard(experiment.id, "id")}
            class="text-ctp-blue hover:text-ctp-blue/80 transition-colors flex items-center gap-1 text-xs font-mono"
          >
            <span class="truncate max-w-24 sm:max-w-32">{experiment.id}</span>
            {#if copiedId}
              <ClipboardCheck size={10} class="text-ctp-green" />
            {:else}
              <Copy size={10} />
            {/if}
          </button>
        </div>

        <!-- Status -->
        <div class="space-y-1">
          <div class="text-xs text-ctp-subtext0 flex items-center gap-1">
            <Activity size={10} />
            status
          </div>
          <div class="text-xs font-mono">
            <span
              class="text-ctp-{experiment.status === 'COMPLETED'
                ? 'green'
                : experiment.status === 'RUNNING'
                  ? 'yellow'
                  : experiment.status === 'FAILED'
                    ? 'red'
                    : 'subtext0'}"
            >
              {experiment.status?.toLowerCase() || "unknown"}
            </span>
          </div>
        </div>

        <!-- Duration -->
        {#if experiment.startedAt}
          <div class="space-y-1">
            <div class="text-xs text-ctp-subtext0 flex items-center gap-1">
              <Clock size={10} />
              duration
            </div>
            <div class="text-xs text-ctp-text font-mono">
              {experiment.endedAt
                ? `${Math.round((new Date(experiment.endedAt).getTime() - new Date(experiment.startedAt).getTime()) / 1000)}s`
                : experiment.status === "RUNNING"
                  ? `${Math.round((Date.now() - new Date(experiment.startedAt).getTime()) / 1000)}s (running)`
                  : "n/a"}
            </div>
          </div>
        {/if}
      </div>

      <!-- Description -->
      {#if experiment.description}
        <div class="pl-4 md:pl-6 space-y-1">
          <div class="text-xs text-ctp-subtext0">description</div>
          <div class="text-sm text-ctp-subtext1 break-words">
            {experiment.description}
          </div>
        </div>
      {/if}
    </div>

    <!-- Tags -->
    {#if experiment.tags && experiment.tags.length > 0}
      <div class="space-y-2">
        <div class="flex items-center gap-2">
          <div class="text-sm text-ctp-text">tags</div>
          <div class="text-xs text-ctp-subtext0 font-mono">
            [{experiment.tags.length}]
          </div>
        </div>
        <div class="flex flex-wrap gap-1">
          {#each visibleTags as tag}
            <span
              class="text-xs bg-ctp-blue/20 text-ctp-blue border border-ctp-blue/30 px-2 py-0.5 font-mono"
            >
              {tag}
            </span>
          {/each}
          {#if experiment.tags.length > initialLimit}
            <button
              onclick={() => (showAllTags = !showAllTags)}
              class="text-xs text-ctp-subtext0 hover:text-ctp-blue transition-colors px-2 py-0.5"
            >
              {showAllTags
                ? "less"
                : `+${experiment.tags.length - initialLimit}`}
            </button>
          {/if}
        </div>
      </div>
    {/if}

    <!-- Hyperparameters -->
    {#if experiment.hyperparams && experiment.hyperparams.length > 0}
      <div class="space-y-2">
        <div class="flex items-center gap-2">
          <div class="text-sm text-ctp-text">hyperparameters</div>
          <div class="text-xs text-ctp-subtext0 font-mono">
            [{experiment.hyperparams.length}]
          </div>
        </div>
        <div class="bg-ctp-surface0/10 border border-ctp-surface0/20">
          <div class="grid grid-cols-1 lg:grid-cols-2 gap-0">
            {#each visibleParams as param}
              <div
                class="flex flex-col sm:flex-row sm:items-center sm:justify-between border-b border-ctp-surface0/10 hover:bg-ctp-surface0/20 px-3 py-2 transition-colors text-xs gap-1 sm:gap-2"
              >
                <span class="text-ctp-subtext0 font-mono truncate"
                  >{param.key}</span
                >
                <div class="flex items-center gap-2">
                  <span
                    class="text-ctp-blue font-mono bg-ctp-surface0/20 border border-ctp-surface0/30 px-2 py-1 max-w-24 sm:max-w-32 truncate"
                    title={String(param.value)}
                  >
                    {param.value}
                  </span>
                  <button
                    onclick={() =>
                      copyToClipboard(String(param.value), "param", param.key)}
                    class="text-ctp-subtext0 hover:text-ctp-text transition-colors"
                  >
                    {#if copiedParam === param.key}
                      <ClipboardCheck size={10} class="text-ctp-green" />
                    {:else}
                      <Copy size={10} />
                    {/if}
                  </button>
                </div>
              </div>
            {/each}
          </div>
          {#if experiment.hyperparams.length > initialLimit}
            <button
              onclick={() => (showAllParams = !showAllParams)}
              class="w-full text-xs text-ctp-subtext0 hover:text-ctp-text px-3 py-2 text-center border-t border-ctp-surface0/20 transition-colors"
            >
              {showAllParams
                ? "show less"
                : `show ${experiment.hyperparams.length - initialLimit} more`}
            </button>
          {/if}
        </div>
      </div>
    {/if}

    <!-- Metrics Section with Toggle -->
    {#if scalarMetrics.length > 0 || timeSeriesNames.length > 0}
      <div class="space-y-2">
        <div class="flex items-center gap-4">
          <div class="text-sm text-ctp-text">metrics</div>
          <div class="text-xs text-ctp-subtext0 font-mono">
            [{scalarMetrics.length + timeSeriesNames.length}]
          </div>
          <div class="flex items-center gap-1">
            <button
              onclick={() => (metricsView = "chart")}
              class="text-xs text-ctp-{metricsView === 'chart'
                ? 'blue'
                : 'subtext0'} hover:text-ctp-blue transition-colors"
            >
              [chart]
            </button>
            <button
              onclick={() => (metricsView = "data")}
              class="text-xs text-ctp-{metricsView === 'data'
                ? 'blue'
                : 'subtext0'} hover:text-ctp-blue transition-colors"
            >
              [data]
            </button>
          </div>
        </div>
      </div>
    {/if}

    <!-- Chart View -->
    {#if metricsView === "chart" && timeSeriesNames.length > 0}
      <div class="space-y-3">
        <!-- Interactive chart -->
        <div
          class="bg-ctp-surface0/10 border border-ctp-surface0/20 p-2 md:p-4"
        >
          <InteractiveChart experiment={experimentWithMetrics} />
        </div>
      </div>
    {/if}

    <!-- Data View - Scalar Metrics -->
    {#if metricsView === "data" && scalarMetrics.length > 0}
      <div class="space-y-2">
        <div class="flex items-center gap-2">
          <div class="text-sm text-ctp-text">scalar metrics</div>
          <div class="text-xs text-ctp-subtext0 font-mono">
            [{scalarMetrics.length}]
          </div>
        </div>
        <div class="bg-ctp-surface0/10 border border-ctp-surface0/20">
          <!-- Desktop table -->
          <div class="hidden md:block">
            <div
              class="flex text-xs text-ctp-subtext0 p-3 border-b border-ctp-surface0/20 sticky top-0"
            >
              <div class="w-4">•</div>
              <div class="flex-1">metric</div>
              <div class="w-20 text-right">value</div>
              <div class="w-8"></div>
            </div>

            <div
              class="{showAllScalarMetrics ? '' : 'max-h-60'} overflow-y-auto"
            >
              {#each visibleScalarMetrics as metric}
                <div
                  class="flex text-xs hover:bg-ctp-surface0/20 p-3 transition-colors border-b border-ctp-surface0/5"
                >
                  <div class="w-4 text-ctp-green">●</div>
                  <div
                    class="flex-1 text-ctp-text truncate"
                    title={metric.name}
                  >
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
                  <div class="w-8">
                    <button
                      onclick={() =>
                        copyToClipboard(
                          String(metric.value),
                          "metric",
                          metric.name,
                        )}
                      class="text-ctp-subtext0 hover:text-ctp-text transition-colors"
                    >
                      {#if copiedMetric === metric.name}
                        <ClipboardCheck size={10} class="text-ctp-green" />
                      {:else}
                        <Copy size={10} />
                      {/if}
                    </button>
                  </div>
                </div>
              {/each}
            </div>
          </div>

          <!-- Mobile card layout -->
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
                    <div class="text-ctp-green text-xs">●</div>
                    <div
                      class="text-xs text-ctp-text font-mono truncate"
                      title={metric.name}
                    >
                      {metric.name}
                    </div>
                  </div>
                  <button
                    onclick={() =>
                      copyToClipboard(
                        String(metric.value),
                        "metric",
                        metric.name,
                      )}
                    class="text-ctp-subtext0 hover:text-ctp-text transition-colors"
                  >
                    {#if copiedMetric === metric.name}
                      <ClipboardCheck size={10} class="text-ctp-green" />
                    {:else}
                      <Copy size={10} />
                    {/if}
                  </button>
                </div>
                <div class="flex items-center justify-between text-xs">
                  <div
                    class="text-ctp-blue font-mono"
                    title={String(metric.value)}
                  >
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
              class="w-full text-xs text-ctp-subtext0 hover:text-ctp-text p-3 text-center border-t border-ctp-surface0/20 transition-colors"
            >
              {showAllScalarMetrics
                ? "show less"
                : `show ${scalarMetrics.length - initialLimit} more`}
            </button>
          {/if}
        </div>
      </div>
    {/if}

    <!-- Files and artifacts -->
    {#if files.length > 0}
      <div class="space-y-2">
        <div class="flex items-center gap-2">
          <div class="text-sm text-ctp-text">files & artifacts</div>
          <div class="text-xs text-ctp-subtext0 font-mono">
            [{files.length}]
          </div>
        </div>
        <div class="bg-ctp-surface0/10 border border-ctp-surface0/20">
          {#each files as file}
            <div
              class="flex flex-col sm:flex-row sm:items-center hover:bg-ctp-surface0/20 px-3 py-2 transition-colors text-xs border-b border-ctp-surface0/5 last:border-b-0 gap-2 sm:gap-0"
            >
              <div class="flex items-center gap-2 flex-1 min-w-0">
                {#each [getFileIcon(file.name)] as IconComponent}
                  <IconComponent
                    size={12}
                    class="text-ctp-subtext0 flex-shrink-0"
                  />
                {/each}
                <span class="text-ctp-text truncate" title={file.path}>
                  {file.name}
                </span>
                {#if file.size}
                  <span class="text-ctp-subtext0 font-mono text-xs">
                    {formatFileSize(file.size)}
                  </span>
                {/if}
              </div>
              <div class="flex items-center gap-2 ml-6 sm:ml-2">
                {#if file.url}
                  <a
                    href={file.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    class="text-ctp-subtext0 hover:text-ctp-blue transition-colors flex items-center gap-1"
                    title="View file"
                  >
                    <ExternalLink size={10} />
                    <span class="sm:hidden text-xs">view</span>
                  </a>
                {/if}
                {#if file.downloadUrl}
                  <a
                    href={file.downloadUrl}
                    download={file.name}
                    class="text-ctp-subtext0 hover:text-ctp-green transition-colors flex items-center gap-1"
                    title="Download file"
                  >
                    <Download size={10} />
                    <span class="sm:hidden text-xs">download</span>
                  </a>
                {/if}
              </div>
            </div>
          {/each}
        </div>
      </div>
    {/if}

    <!-- System information -->
    <div class="space-y-2">
      <div class="text-sm text-ctp-text">system info</div>
      <div
        class="bg-ctp-surface0/10 border border-ctp-surface0/20 p-3 space-y-2 text-xs font-mono"
      >
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-3 md:gap-4">
          {#if experiment.createdBy}
            <div class="flex items-center gap-2">
              <User size={10} class="text-ctp-subtext0" />
              <span class="text-ctp-subtext0">created_by:</span>
              <span class="text-ctp-text">{experiment.createdBy}</span>
            </div>
          {/if}

          <div class="flex items-center gap-2">
            <Calendar size={10} class="text-ctp-subtext0" />
            <span class="text-ctp-subtext0">created_at:</span>
            <span class="text-ctp-text">
              {new Date(experiment.createdAt).toISOString()}
            </span>
          </div>

          {#if experiment.updatedAt}
            <div class="flex items-center gap-2">
              <Calendar size={10} class="text-ctp-subtext0" />
              <span class="text-ctp-subtext0">updated_at:</span>
              <span class="text-ctp-text">
                {new Date(experiment.updatedAt).toISOString()}
              </span>
            </div>
          {/if}

          {#if experiment.version}
            <div class="flex items-center gap-2">
              <Hash size={10} class="text-ctp-subtext0" />
              <span class="text-ctp-subtext0">version:</span>
              <span class="text-ctp-text">{experiment.version}</span>
            </div>
          {/if}
        </div>
      </div>
    </div>
  </div>
</div>
