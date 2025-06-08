<script lang="ts">
  import type { Experiment, Metric } from "$lib/types";
  import Chart from "chart.js/auto";
  import { ChartLine, Plus, EyeOff } from "lucide-svelte";
  import { onMount, onDestroy } from "svelte";

  let chartInstance: Chart | null = null;
  let chartCanvas: HTMLCanvasElement | null = $state(null);
  let { experiment }: { experiment: Experiment } = $props();
  let isLoading: boolean = $state(false);
  let selectedMetrics = $state<string[]>([]);
  let metricsData = $state<
    Record<string, { steps: number[]; values: number[] }>
  >({});
  let currentTheme = $state<"light" | "dark">("dark");

  const chartColorKeys = [
    "red", "blue", "green", "yellow", "mauve", "pink", 
    "peach", "teal", "sky", "sapphire", "lavender", "maroon"
  ];

  function getChartColors() {
    const computedStyles = getComputedStyle(document.documentElement);
    return chartColorKeys.map(colorKey => {
      const baseColor = computedStyles.getPropertyValue(`--color-ctp-${colorKey}`).trim();
      return {
        border: baseColor,
        bg: `${baseColor}10`, // hex transparency for ~6% opacity
        point: baseColor
      };
    });
  }

  function getThemeUI() {
    const computedStyles = getComputedStyle(document.documentElement);
    return {
      text: computedStyles.getPropertyValue("--color-ctp-text").trim(),
      crust: computedStyles.getPropertyValue("--color-ctp-crust").trim(),
      mantle: computedStyles.getPropertyValue("--color-ctp-mantle").trim(),
      base: computedStyles.getPropertyValue("--color-ctp-base").trim(),
      overlay0: computedStyles.getPropertyValue("--color-ctp-overlay0").trim(),
      gridLines: `${computedStyles.getPropertyValue("--color-ctp-overlay0").trim()}15`, // hex transparency
    };
  }

  onMount(() => {
    updateTheme();

    window.addEventListener("storage", handleStorageChange);

    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (
          mutation.attributeName === "class" &&
          mutation.target === document.documentElement
        ) {
          updateTheme();
        }
      });
    });

    observer.observe(document.documentElement, { attributes: true });

    return () => {
      observer.disconnect();
      window.removeEventListener("storage", handleStorageChange);
    };
  });

  onDestroy(() => {
    destroyChart();
  });

  function handleStorageChange(event: StorageEvent) {
    if (event.key === "theme") {
      updateTheme();
    }
  }

  function updateTheme() {
    const isDark = document.documentElement.classList.contains("dark");
    currentTheme = isDark ? "dark" : "light";
    if (chartInstance) {
      updateChart();
    }
  }

  async function loadMetrics() {
    try {
      isLoading = true;
      const response = await fetch(`/api/experiments/${experiment.id}/metrics`);
      if (!response.ok) {
        throw new Error(`Failed to load metrics: ${response.statusText}`);
      }
      return await response.json();
    } catch (e) {
      console.error("Error loading metrics:", e);
      return null;
    } finally {
      isLoading = false;
    }
  }

  function destroyChart() {
    if (chartInstance) {
      chartInstance.destroy();
      chartInstance = null;
    }
  }

  function updateChart() {
    destroyChart();
    if (!chartCanvas || selectedMetrics.length === 0) return;

    try {
      const colors = getChartColors();
      const ui = getThemeUI();

      const datasets = selectedMetrics.map((metric, index) => {
        const colorIndex = index % colors.length;
        const color = colors[colorIndex];
        const steps = metricsData[metric]?.steps || [];
        const values = metricsData[metric]?.values || [];
        const dataPoints = steps.map((step, i) => ({ x: step, y: values[i] }));

        return {
          label: metric,
          data: dataPoints,
          borderColor: color.border,
          backgroundColor: color.bg,
          fill: false,
          pointBackgroundColor: color.point,
          pointBorderColor: ui.mantle,
          pointHoverBackgroundColor: 
            getComputedStyle(document.documentElement).getPropertyValue("--color-ctp-mauve").trim(),
          pointHoverBorderColor: ui.base,
          borderWidth: 1.5,
          tension: 0.3,
          pointRadius: 2,
        };
      });

      chartInstance = new Chart(chartCanvas, {
        type: "line",
        data: {
          datasets: datasets,
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          parsing: false,
          normalized: true,
          interaction: {
            mode: "nearest",
            intersect: false,
            axis: "x",
          },
          plugins: {
            decimation: {
              enabled: true,
              algorithm: 'lttb', // Largest-Triangle-Three-Buckets algorithm
              samples: 50, // Maximum number of points to display
            },
            legend: {
              display: true,
              position: "top",
              labels: {
                color: ui.text,
                usePointStyle: true,
                pointStyle: "circle",
                padding: 15,
                font: {
                  size: 11,
                },
              },
            },
            tooltip: {
              backgroundColor: ui.crust,
              titleColor: getComputedStyle(document.documentElement).getPropertyValue("--color-ctp-sky").trim(),
              bodyColor: ui.text,
              borderColor: ui.overlay0,
              position: "nearest",
              caretPadding: 10,
              callbacks: {
                title: function (tooltipItems) {
                  return `Step ${tooltipItems[0].parsed.x}`;
                },
              },
            },
          },
          scales: {
            x: {
              type: "linear",
              position: "bottom",
              title: {
                display: true,
                text: "Step",
                color: ui.text,
              },
              ticks: {
                color: ui.text,
              },
              grid: {
                color: ui.gridLines,
              },
            },
            y: {
              type: "logarithmic",
              position: "left",
              title: {
                display: true,
                text: "Value (log)",
                color: ui.text,
              },
              ticks: {
                color: ui.text,
              },
              grid: {
                color: ui.gridLines,
              },
            },
          },
        },
      });
    } catch (error) {
      console.error("Failed to create chart:", error);
    }
  }

  async function toggleMetric(metric: string) {
    isLoading = true;

    try {
      if (selectedMetrics.includes(metric)) {
        selectedMetrics = selectedMetrics.filter((m) => m !== metric);
      } else {
        selectedMetrics = [...selectedMetrics, metric];
        if (!metricsData[metric]) {
          const metrics = (await loadMetrics()) as Metric[];
          const metricsByName = Object.groupBy(metrics, ({ name }) => name);
          const chart_targets = metricsByName[metric];
          if (chart_targets && chart_targets.length > 0) {
            chart_targets.sort((a, b) => (a.step ?? 0) - (b.step ?? 0));
            const steps = chart_targets.map((l, index) =>
              l.step !== undefined ? l.step : index,
            );
            const values = chart_targets.map((l) =>
              typeof l.value === "number" ? l.value : parseFloat(l.value) || 0,
            );

            metricsData[metric] = { steps, values };
          }
        }
      }

      updateChart();
    } catch (error) {
      console.error("Error toggling metric:", error);
    } finally {
      isLoading = false;
    }
  }

  function resetChart() {
    selectedMetrics = [];
    metricsData = {};
    destroyChart();
  }
</script>

<div class="p-3 sm:p-4 space-y-4 w-full">
  <!-- Available Metrics Header Section -->
  {#if experiment.availableMetrics && experiment.availableMetrics.length > 0}
    <div
      class="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-2 mb-2"
    >
      <h3 class="text-sm font-medium text-ctp-subtext0">Available Metrics</h3>
      {#if selectedMetrics.length > 0}
        <button
          class="self-end sm:self-auto px-2 py-1 text-xs text-ctp-red border border-ctp-red rounded hover:bg-ctp-red/10 transition-colors"
          onclick={resetChart}
        >
          Clear All
        </button>
      {/if}
    </div>

    <!-- Available Metrics Selection Buttons -->
    <div class="flex flex-wrap gap-2 mb-4 max-w-full overflow-x-auto pb-1">
      {#each experiment.availableMetrics as metric}
        <button
          class={`flex items-center gap-1.5 py-1 sm:py-1.5 px-2 sm:px-3 text-xs sm:text-sm font-medium rounded-md transition-colors whitespace-nowrap ${
            selectedMetrics.includes(metric)
              ? "bg-ctp-mauve text-ctp-crust hover:bg-ctp-lavender"
              : "bg-ctp-surface0 text-ctp-text border border-ctp-surface1 hover:bg-ctp-blue hover:text-ctp-crust hover:border-ctp-blue"
          }`}
          onclick={() => toggleMetric(metric)}
        >
          {#if selectedMetrics.includes(metric)}
            <EyeOff size={12} class="sm:hidden" />
            <EyeOff size={14} class="hidden sm:inline" />
          {:else}
            <Plus size={12} class="sm:hidden" />
            <Plus size={14} class="hidden sm:inline" />
          {/if}
          {metric}
        </button>
      {/each}
    </div>
  {/if}

  <!-- Chart Display Section -->
  {#if selectedMetrics.length > 0}
    <div
      class="relative h-60 sm:h-80 w-full rounded-md border border-ctp-surface1 bg-ctp-mantle overflow-hidden shadow-md"
    >
      {#if isLoading}
        <div
          class="absolute inset-0 flex items-center justify-center bg-ctp-mantle/80 backdrop-blur-sm z-10"
        >
          <div class="animate-pulse text-[#91d7e3]">Loading data...</div>
        </div>
      {/if}
      <div class="absolute inset-0 p-2 sm:p-4">
        <canvas bind:this={chartCanvas}></canvas>
      </div>
    </div>
    <!-- Empty State -->
  {:else if experiment.availableMetrics && experiment.availableMetrics.length > 0}
    <div
      class="flex flex-col items-center justify-center h-60 sm:h-80 w-full rounded-md border border-ctp-surface1 bg-ctp-mantle p-4 sm:p-8"
    >
      <ChartLine size={24} class="text-ctp-overlay0 mb-3 sm:mb-4 sm:text-3xl" />
      <p class="text-ctp-subtext0 text-xs sm:text-sm text-center max-w-md">
        Select metrics from above to view and compare chart data
      </p>
    </div>
  {/if}
</div>

<style>
  canvas {
    background-color: transparent;
    border-radius: 4px;
  }
</style>
