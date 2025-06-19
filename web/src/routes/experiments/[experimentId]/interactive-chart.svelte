<script lang="ts">
  import type { Experiment } from "$lib/types";
  import Chart from "chart.js/auto";
  import { ChartLine, ChevronDown } from "lucide-svelte";
  import { onMount, onDestroy } from "svelte";
  import { startTimer } from "$lib/utils/timing";
  import { browser } from "$app/environment";

  const MAX_DATA_POINTS_TO_RENDER = 100;
  const CHART_COLOR_KEYS = [
    "red",
    "blue",
    "green",
    "yellow",
    "mauve",
    "pink",
    "peach",
    "teal",
    "sky",
    "sapphire",
    "lavender",
    "maroon",
  ];

  let {
    experiment,
    selectedMetrics = $bindable([]),
  }: {
    experiment: ExperimentWithMetricData;
    selectedMetrics?: string[];
  } = $props();

  let chartInstance: Chart | null = null;
  let chartCanvas: HTMLCanvasElement | null = $state(null);
  let detailsElement: HTMLDetailsElement | null = $state(null);

  interface ExperimentWithMetricData extends Experiment {
    metricData?: Record<string, number[]>;
  }

  let metricsData = $state<
    Record<string, { steps: number[]; values: number[] }>
  >({});
  let searchFilter = $state<string>("");

  let chartTheme = $state(getTheme());

  const availableMetrics = $derived.by(() => experiment.availableMetrics || []);
  const filteredMetrics = $derived.by(() =>
    availableMetrics.filter((metric) =>
      metric.toLowerCase().includes(searchFilter.toLowerCase()),
    ),
  );

  const chartOptions = $derived.by(() => getChartOptions(chartTheme));

  function selectAllMetrics() {
    selectedMetrics = [...availableMetrics];
  }

  function clearAllMetrics() {
    selectedMetrics = [];
  }

  function toggleMetricCheckbox(metric: string) {
    if (selectedMetrics.includes(metric)) {
      selectedMetrics = selectedMetrics.filter((m) => m !== metric);
    } else {
      selectedMetrics = [...selectedMetrics, metric];
    }
  }

  function getTheme() {
    if (!browser) {
      return {
        colors: [],
        text: "#000",
        mantle: "#fff",
        overlay0: "#ccc",
        sky: "#007bff",
        fadedGridLines: "#eee",
        axisTicks: "#333",
      };
    }

    const computedStyles = getComputedStyle(document.documentElement);
    const colors = CHART_COLOR_KEYS.map((key) =>
      computedStyles.getPropertyValue(`--color-ctp-${key}`).trim(),
    );
    return {
      colors,
      text: computedStyles.getPropertyValue("--color-ctp-text").trim(),
      mantle: computedStyles.getPropertyValue("--color-ctp-mantle").trim(),
      overlay0: computedStyles.getPropertyValue("--color-ctp-overlay0").trim(),
      sky: computedStyles.getPropertyValue("--color-ctp-sky").trim(),
      fadedGridLines:
        computedStyles.getPropertyValue("--color-ctp-surface1").trim() + "33",
      axisTicks: computedStyles.getPropertyValue("--color-ctp-subtext0").trim(),
    };
  }

  function getChartOptions(ui: ReturnType<typeof getTheme>) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      parsing: false as const,
      normalized: true,
      interaction: {
        mode: "nearest" as const,
        intersect: false,
        axis: "x" as const,
      },
      plugins: {
        legend: {
          display: true,
          position: "top" as const,
          labels: {
            color: ui.text,
            usePointStyle: true,
            pointStyle: "circle",
            padding: 15,
            font: { size: 11 },
          },
        },
        tooltip: {
          enabled: true,
          backgroundColor: ui.mantle + "cc",
          titleColor: ui.sky,
          bodyColor: ui.text,
          borderColor: ui.overlay0 + "33",
          borderWidth: 1,
          position: "nearest" as const,
          caretPadding: 12,
          cornerRadius: 8,
          displayColors: true,
          titleFont: { size: 13, weight: "bold" as const },
          bodyFont: { size: 12 },
          padding: 12,
          callbacks: {
            title: (tooltipItems: any) => `Step ${tooltipItems[0].parsed.x}`,
            label: (context: any) =>
              `${context.dataset.label}: ${context.parsed.y.toFixed(4)}`,
          },
        },
      },
      scales: {
        x: {
          type: "linear" as const,
          title: { display: true, text: "Step", color: ui.axisTicks },
          ticks: { color: ui.axisTicks },
          grid: { color: ui.fadedGridLines },
        },
        y: {
          type: "logarithmic" as const,
          title: { display: true, text: "Value (log)", color: ui.axisTicks },
          ticks: { color: ui.axisTicks },
          grid: { color: ui.fadedGridLines },
        },
      },
      onHover: (event: any, activeElements: any[]) => {
        (event.native.target as HTMLElement).style.cursor =
          activeElements.length > 0 ? "pointer" : "default";
      },
    };
  }

  function createOrUpdateChart() {
    const timer = startTimer("chart.createOrUpdate", {
      experimentId: experiment.id.toString(),
    });

    if (!chartCanvas || selectedMetrics.length === 0) {
      destroyChart();
      timer.end({ skipped: "true" });
      return;
    }

    try {
      const datasets = selectedMetrics.map((metric, index) => {
        const color = chartTheme.colors[index % chartTheme.colors.length];
        const metricData = metricsData[metric] || { steps: [], values: [] };

        const rawDataPoints = metricData.steps.map((step, i) => ({
          x: step,
          y: metricData.values[i],
        }));

        const data =
          rawDataPoints.length > MAX_DATA_POINTS_TO_RENDER
            ? rawDataPoints.filter(
                (_, i) =>
                  i %
                    Math.ceil(
                      rawDataPoints.length / MAX_DATA_POINTS_TO_RENDER,
                    ) ===
                  0,
              )
            : rawDataPoints;

        return {
          label: metric,
          data,
          borderColor: color,
          backgroundColor: color + "30",
          fill: true,
          pointRadius: 0,
          pointHoverRadius: 0,
          borderWidth: 2,
          tension: 0.2,
        };
      });

      if (chartInstance) {
        chartInstance.data.datasets = datasets;
        chartInstance.options = chartOptions;
        chartInstance.update();
      } else {
        chartInstance = new Chart(chartCanvas, {
          type: "line",
          data: { datasets },
          options: chartOptions,
        });
      }
      timer.end({ success: "true" });
    } catch (error) {
      console.error("Failed to create or update chart:", error);
      timer.end({
        error: error instanceof Error ? error.message : "Unknown error",
      });
    }
  }

  function destroyChart() {
    if (chartInstance) {
      chartInstance.destroy();
      chartInstance = null;
    }
  }

  function downloadChart() {
    if (chartCanvas) {
      const link = document.createElement("a");
      link.href = chartCanvas.toDataURL("image/png");
      link.download = `${experiment.id}-chart.png`;
      link.click();
    }
  }

  function handleKeydown(event: KeyboardEvent) {
    if (
      event.target instanceof HTMLInputElement ||
      event.target instanceof HTMLTextAreaElement
    ) {
      return;
    }

    switch (event.key) {
      case "m":
        if (detailsElement) {
          event.preventDefault();
          detailsElement.open = !detailsElement.open;
        }
        break;

      case "/":
        if (detailsElement?.open) {
          event.preventDefault();
          const searchInput = detailsElement.querySelector<HTMLInputElement>(
            'input[type="search"]',
          );
          searchInput?.focus();
        }
        break;

      case "Escape":
        if (detailsElement?.open) {
          event.preventDefault();
          detailsElement.open = false;
        }
        break;
    }
  }

  onMount(() => {
    const handleThemeChange = () => {
      chartTheme = getTheme();
    };

    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    mediaQuery.addEventListener("change", handleThemeChange);

    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (
          mutation.attributeName === "class" &&
          mutation.target === document.documentElement
        ) {
          const classList = (mutation.target as Element).classList;
          if (classList.contains("dark") || classList.contains("light")) {
            handleThemeChange();
          }
        }
      });
    });

    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    });

    window.addEventListener("keydown", handleKeydown);

    return () => {
      mediaQuery.removeEventListener("change", handleThemeChange);
      window.removeEventListener("keydown", handleKeydown);
      observer.disconnect();
      destroyChart();
    };
  });

  $effect(() => {
    const el = detailsElement;
    if (!el) return;

    const handleClickOutside = (event: MouseEvent) => {
      if (el && !el.contains(event.target as Node)) {
        el.open = false;
      }
    };

    document.addEventListener("click", handleClickOutside);
    return () => document.removeEventListener("click", handleClickOutside);
  });

  $effect(() => {
    for (const metric of selectedMetrics) {
      if (!metricsData[metric]) {
        const values = experiment.metricData?.[metric] || [];
        // Ensure values are positive for log scale. Replace 0 or negative with a small number.
        const sanitizedValues = values.map((v) => (v > 0 ? v : 1e-9));
        metricsData[metric] = {
          steps: sanitizedValues.map((_, i) => i),
          values: sanitizedValues,
        };
      }
    }
    createOrUpdateChart();
  });

  onDestroy(destroyChart);
</script>

<div class="w-full">
  <!-- Metric Selector -->
  {#if availableMetrics.length > 0}
    <div class="mb-4">
      <details class="relative" bind:this={detailsElement}>
        <summary
          class="flex items-center justify-between cursor-pointer p-2 md:p-3 bg-ctp-surface0/20 border border-ctp-surface0/30 hover:bg-ctp-surface0/30 transition-colors text-xs md:text-sm"
        >
          <span class="text-ctp-text">
            select metrics ({selectedMetrics.length}/{availableMetrics.length})
          </span>
          <ChevronDown size={12} class="text-ctp-subtext0 md:w-4 md:h-4" />
        </summary>

        <div
          class="absolute top-full left-0 right-0 mt-1 bg-ctp-mantle border border-ctp-surface0/30 shadow-lg z-10 max-h-60 overflow-y-auto"
        >
          <!-- Search filter -->
          <div class="p-2 border-b border-ctp-surface0/20">
            <input
              type="search"
              placeholder="filter metrics..."
              bind:value={searchFilter}
              class="w-full bg-ctp-surface0/20 border border-ctp-surface0/30 px-2 py-1 text-ctp-text placeholder-ctp-subtext0 focus:outline-none focus:ring-1 focus:ring-ctp-blue focus:border-ctp-blue transition-all text-xs"
            />
          </div>

          <!-- Control buttons -->
          <div class="flex gap-2 p-2 border-b border-ctp-surface0/20">
            <button
              onclick={selectAllMetrics}
              class="px-2 py-1 text-xs bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-green hover:bg-ctp-green/10 hover:border-ctp-green/30 transition-all"
            >
              all
            </button>
            <button
              onclick={clearAllMetrics}
              class="px-2 py-1 text-xs bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-red hover:bg-ctp-red/10 hover:border-ctp-red/30 transition-all"
            >
              clear
            </button>
          </div>

          <!-- Metric checkboxes -->
          <div class="p-1">
            {#each filteredMetrics as metric}
              <label
                class="flex items-center gap-2 p-1 hover:bg-ctp-surface0/20 cursor-pointer text-xs"
              >
                <input
                  type="checkbox"
                  checked={selectedMetrics.includes(metric)}
                  onchange={() => toggleMetricCheckbox(metric)}
                  class="text-ctp-blue focus:ring-ctp-blue focus:ring-1 w-3 h-3"
                />
                <span class="text-ctp-text">{metric}</span>
              </label>
            {/each}

            {#if filteredMetrics.length === 0}
              <div class="p-2 text-xs text-ctp-subtext0 text-center">
                no metrics found
              </div>
            {/if}
          </div>
        </div>
      </details>
    </div>
  {/if}

  <!-- Chart Display Section -->
  {#if selectedMetrics.length > 0}
    <div
      class="relative h-60 sm:h-80 w-full border border-ctp-surface0/20 bg-transparent overflow-hidden"
    >
      <div class="absolute inset-0">
        <canvas bind:this={chartCanvas} class="chart-canvas"></canvas>
      </div>
      <button
        class="absolute top-2 right-2 text-xs text-ctp-subtext0 hover:text-ctp-blue transition-colors bg-ctp-surface0/20 border border-ctp-surface0/30 px-2 py-1"
        onclick={downloadChart}
      >
        [png]
      </button>
    </div>
    <!-- Empty State -->
  {:else if experiment.availableMetrics && experiment.availableMetrics.length > 0}
    <div
      class="flex flex-col items-center justify-center h-60 sm:h-80 w-full border border-ctp-surface0/20"
    >
      <ChartLine
        size={20}
        class="text-ctp-subtext0 mb-2 sm:mb-3 sm:w-6 sm:h-6"
      />
      <p class="text-ctp-subtext0 text-xs text-center max-w-md">
        $ select metrics to view chart data
      </p>
    </div>
  {/if}
</div>

<style>
  .chart-canvas {
    background-color: transparent;
    border-radius: 4px;
    touch-action: none;
    user-select: none;
    -webkit-user-select: none;
    -webkit-touch-callout: none;
  }
</style>
