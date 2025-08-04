<script lang="ts">
  import Chart from "chart.js/auto";
  import { ChartLine } from "@lucide/svelte";
  import { onMount, onDestroy } from "svelte";
  import { startTimer } from "$lib/utils/timing";
  import { browser } from "$app/environment";
  import { SearchDropdown } from "$lib/components";

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

  let { metricData, availableMetrics } = $props();
  let chartInstance: Chart | null = null;
  let chartCanvas: HTMLCanvasElement | null = $state(null);
  let selectedMetrics = $state<string[]>([]);

  let searchFilter = $state<string>("");

  let chartTheme = $state(getTheme());

  const chartOptions = $derived.by(() => getChartOptions(chartTheme));

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
      // experimentId: experiment.id.toString(), // Removed as experiment is no longer a prop
    });

    if (!chartCanvas || selectedMetrics.length === 0) {
      destroyChart();
      timer.end({ skipped: "true" });
      return;
    }

    try {
      const datasets = selectedMetrics.map((metric, index) => {
        const color = chartTheme.colors[index % chartTheme.colors.length];
        const metricValues = metricData?.[metric] || [];

        const rawDataPoints = metricValues.map((value: number, i: number) => ({
          x: i,
          y: value > 0 ? value : 1e-9, // Sanitize values for logarithmic scale
        }));

        const data =
          rawDataPoints.length > MAX_DATA_POINTS_TO_RENDER
            ? rawDataPoints.filter(
                (_: any, i: number) =>
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
      link.download = `experiment-chart.png`; // Simplified filename
      link.click();
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

    return () => {
      mediaQuery.removeEventListener("change", handleThemeChange);
      observer.disconnect();
      destroyChart();
    };
  });

  $effect(() => {
    // No need to process metricData here, it's passed directly
    createOrUpdateChart();
  });

  onDestroy(destroyChart);
</script>

<div class="w-full">
  <!-- Metric Selector -->
  {#if availableMetrics.length > 0}
    <div class="mb-4">
      <SearchDropdown
        items={availableMetrics}
        bind:selectedItems={selectedMetrics}
        bind:searchQuery={searchFilter}
        getItemText={(metric) => metric}
        itemTypeName="metrics"
        placeholder=""
      />
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
        class="absolute top-2 right-2 text-xs text-ctp-subtext0 hover:ctp-blue transition-colors bg-ctp-surface0/20 border border-ctp-surface0/30 px-2 py-1"
        onclick={downloadChart}
      >
        [png]
      </button>
    </div>
    <!-- Empty State -->
  {:else if availableMetrics && availableMetrics.length > 0}
    <div
      class="flex flex-col items-center justify-center h-60 sm:h-80 w-full border border-ctp-surface0/20"
    >
      <ChartLine
        size={20}
        class="text-ctp-subtext0 mb-2 sm:mb-3 sm:w-6 sm:h-6"
      />
      <p class="text-ctp-subtext0 text-xs text-center max-w-md">
        select metrics to view chart data
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
