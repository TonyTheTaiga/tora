<script lang="ts">
  import type { Experiment, Metric } from "$lib/types";
  import Chart from "chart.js/auto";
  import { ChartLine, Plus, EyeOff } from "lucide-svelte";

  let chartInstance: Chart | null = null;
  let chartCanvas: HTMLCanvasElement | null = $state(null);
  let { experiment }: { experiment: Experiment } = $props();
  let isLoading: boolean = $state(false);
  let selectedMetrics = $state<string[]>([]);
  let metricsData = $state<
    Record<string, { steps: number[]; values: number[] }>
  >({});

  const COLORS = [
    { border: "#74c7ec", bg: "rgba(116, 199, 236, 0.15)", point: "#b4befe" }, // sapphire
    { border: "#f5c2e7", bg: "rgba(245, 194, 231, 0.15)", point: "#f38ba8" }, // pink
    { border: "#a6e3a1", bg: "rgba(166, 227, 161, 0.15)", point: "#94e2d5" }, // green
    { border: "#fab387", bg: "rgba(250, 179, 135, 0.15)", point: "#f9e2af" }, // peach
    { border: "#cba6f7", bg: "rgba(203, 166, 247, 0.15)", point: "#89b4fa" }, // mauve
    { border: "#f38ba8", bg: "rgba(243, 139, 168, 0.15)", point: "#eba0ac" }, // red
  ];

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
      const datasets = selectedMetrics.map((metric, index) => {
        const colorIndex = index % COLORS.length;
        const color = COLORS[colorIndex];
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
          pointBorderColor: "#181825" /* mantle */,
          pointHoverBackgroundColor: "#cba6f7" /* mauve */,
          pointHoverBorderColor: "#1e1e2e" /* base */,
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
                color: "#cdd6f4" /* text */,
                usePointStyle: true,
                pointStyle: "circle",
                padding: 15,
                font: {
                  size: 11,
                },
              },
            },
            tooltip: {
              backgroundColor: "#11111b" /* crust */,
              titleColor: "#74c7ec" /* sapphire */,
              bodyColor: "#cdd6f4" /* text */,
              borderColor: "#6c7086" /* overlay0 */,
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
                color: "#cdd6f4" /* text */,
              },
              ticks: {
                color: "#cdd6f4" /* text */,
              },
              grid: {
                color: "rgba(180, 190, 254, 0.08)",
              },
            },
            y: {
              type: "linear",
              position: "left",
              title: {
                display: true,
                text: "Value",
                color: "#cdd6f4" /* text */,
              },
              ticks: {
                color: "#cdd6f4" /* text */,
              },
              grid: {
                color: "rgba(180, 190, 254, 0.08)",
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

<div class="p-5 space-y-4">
  {#if experiment.availableMetrics && experiment.availableMetrics.length > 0}
    <div class="flex justify-between items-center mb-2">
      <h3 class="text-sm font-medium text-ctp-subtext0">Available Metrics</h3>
      {#if selectedMetrics.length > 0}
        <button
          class="px-2 py-1 text-xs text-ctp-red border border-ctp-red rounded hover:bg-ctp-red/10 transition-colors"
          onclick={resetChart}
        >
          Clear All
        </button>
      {/if}
    </div>

    <div class="flex flex-wrap gap-2 mb-4">
      {#each experiment.availableMetrics as metric}
        <button
          class={`flex items-center gap-1.5 py-1.5 px-3 text-sm font-medium rounded-md transition-colors ${
            selectedMetrics.includes(metric)
              ? "bg-ctp-mauve text-ctp-crust hover:bg-ctp-lavender"
              : "bg-ctp-surface0 text-ctp-text border border-ctp-surface1 hover:bg-ctp-blue hover:text-ctp-crust hover:border-ctp-blue"
          }`}
          onclick={() => toggleMetric(metric)}
        >
          {#if selectedMetrics.includes(metric)}
            <EyeOff size={14} />
          {:else}
            <Plus size={14} />
          {/if}
          {metric}
        </button>
      {/each}
    </div>
  {/if}

  {#if selectedMetrics.length > 0}
    <div
      class="relative h-80 w-full rounded-md border border-ctp-surface1 bg-ctp-mantle overflow-hidden shadow-md"
    >
      {#if isLoading}
        <div
          class="absolute inset-0 flex items-center justify-center bg-ctp-mantle/80 backdrop-blur-sm z-10"
        >
          <div class="animate-pulse text-[#89dceb]">Loading data...</div>
        </div>
      {/if}
      <div class="absolute inset-0 p-4">
        <canvas bind:this={chartCanvas}></canvas>
      </div>
    </div>
  {:else if experiment.availableMetrics && experiment.availableMetrics.length > 0}
    <div
      class="flex flex-col items-center justify-center h-80 w-full rounded-md border border-ctp-surface1 bg-ctp-mantle p-8"
    >
      <ChartLine size={32} class="text-ctp-overlay0 mb-4" />
      <p class="text-ctp-subtext0 text-sm text-center max-w-md">
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
