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

  // Catppuccin colors
  const darkColors = [
    { border: "#74c7ec", bg: "rgba(116, 199, 236, 0.15)", point: "#b4befe" }, // sapphire
    { border: "#f5c2e7", bg: "rgba(245, 194, 231, 0.15)", point: "#f38ba8" }, // pink
    { border: "#a6e3a1", bg: "rgba(166, 227, 161, 0.15)", point: "#94e2d5" }, // green
    { border: "#fab387", bg: "rgba(250, 179, 135, 0.15)", point: "#f9e2af" }, // peach
    { border: "#cba6f7", bg: "rgba(203, 166, 247, 0.15)", point: "#89b4fa" }, // mauve
    { border: "#f38ba8", bg: "rgba(243, 139, 168, 0.15)", point: "#eba0ac" }, // red
  ];

  const lightColors = [
    { border: "#209fb5", bg: "rgba(32, 159, 181, 0.15)", point: "#7287fd" }, // sapphire
    { border: "#ea76cb", bg: "rgba(234, 118, 203, 0.15)", point: "#d20f39" }, // pink
    { border: "#40a02b", bg: "rgba(64, 160, 43, 0.15)", point: "#179299" }, // green
    { border: "#fe640b", bg: "rgba(254, 100, 11, 0.15)", point: "#df8e1d" }, // peach
    { border: "#8839ef", bg: "rgba(136, 57, 239, 0.15)", point: "#1e66f5" }, // mauve
    { border: "#d20f39", bg: "rgba(210, 15, 57, 0.15)", point: "#e64553" }, // red
  ];

  function getThemeColors() {
    return currentTheme === "dark" ? darkColors : lightColors;
  }

  // Theme UI values for dark mode
  const darkThemeUI = {
    text: "#cdd6f4",
    crust: "#11111b",
    mantle: "#181825",
    base: "#1e1e2e",
    overlay0: "#6c7086",
    gridLines: "rgba(180, 190, 254, 0.08)",
  };

  // Theme UI values for light mode
  const lightThemeUI = {
    text: "#4c4f69",
    crust: "#dce0e8",
    mantle: "#e6e9ef",
    base: "#eff1f5",
    overlay0: "#9ca0b0",
    gridLines: "rgba(114, 135, 253, 0.08)",
  };

  function getThemeUI() {
    return currentTheme === "dark" ? darkThemeUI : lightThemeUI;
  }

  onMount(() => {
    // Check for theme
    updateTheme();
    
    // Listen for theme changes
    window.addEventListener("storage", handleStorageChange);
    
    // Observe class changes on the document element
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
      const colors = getThemeColors();
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
          pointHoverBackgroundColor: currentTheme === "dark" ? "#cba6f7" : "#8839ef",
          pointHoverBorderColor: ui.base,
          borderWidth: 2,
          tension: 0.3,
          pointRadius: 3,
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
          parsing: false, // Disable parsing as we use {x,y} format
          normalized: true,
          interaction: {
            mode: "nearest",
            intersect: false,
            axis: "x",
          },
          plugins: {
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
              titleColor: currentTheme === "dark" ? "#74c7ec" : "#209fb5",
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
              type: "linear",
              position: "left",
              title: {
                display: true,
                text: "Value",
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

  {#if selectedMetrics.length > 0}
    <div
      class="relative h-60 sm:h-80 w-full rounded-md border border-ctp-surface1 bg-ctp-mantle overflow-hidden shadow-md"
    >
      {#if isLoading}
        <div
          class="absolute inset-0 flex items-center justify-center bg-ctp-mantle/80 backdrop-blur-sm z-10"
        >
          <div class="animate-pulse text-[#89dceb]">Loading data...</div>
        </div>
      {/if}
      <div class="absolute inset-0 p-2 sm:p-4">
        <canvas bind:this={chartCanvas}></canvas>
      </div>
    </div>
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
