<script lang="ts">
  import type { Experiment, Metric } from "$lib/types";
  import Chart from "chart.js/auto";
  import { ChartLine, ChevronDown } from "lucide-svelte";
  import { onMount, onDestroy } from "svelte";
  import { startTimer } from "$lib/utils/timing";

  let chartInstance: Chart | null = null;
  let chartCanvas: HTMLCanvasElement | null = $state(null);
  let {
    experiment,
    selectedMetrics = $bindable([]),
  }: {
    experiment: Experiment;
    selectedMetrics?: string[];
  } = $props();
  let isLoading: boolean = $state(false);
  let metricsData = $state<
    Record<string, { steps: number[]; values: number[] }>
  >({});

  let searchFilter = $state<string>("");
  let availableMetrics = $derived.by(() => experiment.availableMetrics || []);
  let filteredMetrics = $derived.by(() =>
    availableMetrics.filter((metric) =>
      metric.toLowerCase().includes(searchFilter.toLowerCase()),
    ),
  );

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

  const chartColorKeys = [
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

  function getChartColors() {
    const computedStyles = getComputedStyle(document.documentElement);
    return chartColorKeys.map((colorKey) => {
      const baseColor = computedStyles
        .getPropertyValue(`--color-ctp-${colorKey}`)
        .trim();
      return {
        border: baseColor,
        bg: `${baseColor}20`,
        point: baseColor,
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
      sky: computedStyles.getPropertyValue("--color-ctp-sky").trim(), // Added for tooltip title
      // gridLines: `${computedStyles.getPropertyValue("--color-ctp-overlay0").trim()}15`, // Old gridLines
      fadedGridLines:
        computedStyles.getPropertyValue("--color-ctp-surface1").trim() + "33",
      axisTicks: computedStyles.getPropertyValue("--color-ctp-subtext0").trim(),
    };
  }

  onMount(() => {
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const handleThemeChange = () => {
      if (chartInstance) {
        updateChart();
      }
    };

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
    };
  });

  // Fallback DOM listener for touchend events (Chart.js plugins don't always receive them reliably)
  $effect(() => {
    const currentCanvas = chartCanvas;

    if (currentCanvas) {
      const clearTooltipOnTouchEnd = () => {
        if (chartInstance && chartInstance.tooltip) {
          if (typeof chartInstance.tooltip.setActiveElements === "function") {
            chartInstance.tooltip.setActiveElements([], { x: 0, y: 0 });
            chartInstance.update("none");
          }
        }
      };

      currentCanvas.addEventListener("touchend", clearTooltipOnTouchEnd);
      currentCanvas.addEventListener("touchcancel", clearTooltipOnTouchEnd);

      return () => {
        currentCanvas.removeEventListener("touchend", clearTooltipOnTouchEnd);
        currentCanvas.removeEventListener(
          "touchcancel",
          clearTooltipOnTouchEnd,
        );
      };
    }
  });

  onDestroy(() => {
    destroyChart();
  });

  async function loadMetrics() {
    const timer = startTimer("chart.loadMetrics", {
      experimentId: experiment.id,
    });
    try {
      isLoading = true;
      const response = await fetch(`/api/experiments/${experiment.id}/metrics`);
      if (!response.ok) {
        throw new Error(`Failed to load metrics: ${response.statusText}`);
      }
      const data = await response.json();
      timer.end({ metricsCount: data.length });
      return data;
    } catch (e) {
      timer.end({ error: e instanceof Error ? e.message : "Unknown error" });
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
    const timer = startTimer("chart.updateChart", {
      experimentId: experiment.id.toString(),
      selectedMetricsCount: selectedMetrics.length.toString(),
    });

    destroyChart();
    if (!chartCanvas || selectedMetrics.length === 0) {
      timer.end({ skipped: "true" });
      return;
    }

    try {
      const colors = getChartColors();
      const ui = getThemeUI();

      const datasets = selectedMetrics.map((metric, index) => {
        const colorIndex = index % colors.length;
        const color = colors[colorIndex];
        const steps = metricsData[metric]?.steps || [];
        const values = metricsData[metric]?.values || [];
        const rawDataPoints = steps.map((step, i) => ({
          x: step,
          y: values[i],
        }));

        const targetSamples = 50;
        const dataPoints =
          rawDataPoints.length > targetSamples
            ? rawDataPoints.filter(
                (_, i) =>
                  i % Math.ceil(rawDataPoints.length / targetSamples) === 0,
              )
            : rawDataPoints;

        return {
          label: metric,
          data: dataPoints,
          borderColor: color.border,
          backgroundColor: color.border + "40", // Updated background color
          fill: true, // Enabled fill
          pointBackgroundColor: color.point,
          pointBorderColor: ui.base, // Updated point border color
          pointHoverBackgroundColor: getComputedStyle(document.documentElement)
            .getPropertyValue("--color-ctp-mauve")
            .trim(),
          pointHoverBorderColor: ui.base,
          borderWidth: 2, // Updated border width
          tension: 0.3,
          pointRadius: 4,
          pointHoverRadius: 8,
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
          events: [
            "mousemove",
            "mouseout",
            "click",
            "touchstart",
            "touchmove",
            "touchend",
          ],
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
              enabled: true,
              backgroundColor: ui.base + "cc", // Updated tooltip background
              titleColor: ui.sky, // Using ui.sky from getThemeUI
              bodyColor: ui.text,
              borderColor: ui.overlay0 + "33", // Updated tooltip border
              borderWidth: 1, // Added border width for tooltip
              position: "nearest",
              caretPadding: 12,
              cornerRadius: 8,
              displayColors: true,
              titleFont: {
                size: 13,
                weight: "bold",
              },
              bodyFont: {
                size: 12,
              },
              padding: 12,
              callbacks: {
                title: function (tooltipItems) {
                  return `Step ${tooltipItems[0].parsed.x}`;
                },
                label: function (context) {
                  const value = context.parsed.y;
                  return `${context.dataset.label}: ${value.toFixed(4)}`;
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
                color: ui.axisTicks, // Updated axis title color
              },
              ticks: {
                color: ui.axisTicks, // Updated axis ticks color
              },
              grid: {
                color: ui.fadedGridLines, // Updated grid line color
              },
            },
            y: {
              type: "logarithmic",
              position: "left",
              title: {
                display: true,
                text: "Value (log)",
                color: ui.axisTicks, // Updated axis title color
              },
              ticks: {
                color: ui.axisTicks, // Updated axis ticks color
              },
              grid: {
                color: ui.fadedGridLines, // Updated grid line color
              },
            },
          },
          onHover: (event, activeElements) => {
            if (event.native) {
              (event.native.target as HTMLElement).style.cursor =
                activeElements.length > 0 ? "pointer" : "default";
            }
          },
        },
        plugins: [
          {
            id: "touchAndTooltipHandler",
            beforeEvent(chart, args) {
              const event = args.event;
              const eventType = event.type as string;

              if (eventType === "touchstart" || eventType === "touchmove") {
                if (event.native) {
                  event.native.preventDefault();
                }
              }

              if (
                event.type === "mouseout" ||
                eventType === "touchend" ||
                eventType === "mouseup"
              ) {
                if (
                  chart.tooltip &&
                  typeof chart.tooltip.setActiveElements === "function"
                ) {
                  chart.tooltip.setActiveElements([], { x: 0, y: 0 });
                  chart.update("none");
                }
              }
            },
            afterEvent(chart, args) {
              const event = args.event;
              const eventType = event.type as string;

              // Additional cleanup for mouse leave
              if (eventType === "mouseleave") {
                if (
                  chart.tooltip &&
                  typeof chart.tooltip.setActiveElements === "function"
                ) {
                  chart.tooltip.setActiveElements([], { x: 0, y: 0 });
                  chart.update("none");
                }
              }
            },
          },
        ],
      });
      timer.end({ success: "true" });
    } catch (error) {
      timer.end({
        error: error instanceof Error ? error.message : "Unknown error",
      });
      console.error("Failed to create chart:", error);
    }
  }

  async function loadMetricData(metric: string) {
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

  $effect(() => {
    if (selectedMetrics.length > 0) {
      const loadAndUpdate = async () => {
        isLoading = true;
        try {
          for (const metric of selectedMetrics) {
            await loadMetricData(metric);
          }
          updateChart();
        } catch (error) {
          console.error("Error loading metric data:", error);
        } finally {
          isLoading = false;
        }
      };
      loadAndUpdate();
    } else {
      destroyChart();
    }
  });
</script>

<div class="w-full">
  <!-- Metric Selector -->
  {#if availableMetrics.length > 0}
    <div class="mb-4">
      <details class="relative">
        <summary
          class="flex items-center justify-between cursor-pointer p-2 md:p-3 bg-ctp-surface0/20 border border-ctp-surface0/30 hover:bg-ctp-surface0/30 transition-colors text-xs md:text-sm"
        >
          <span class="text-ctp-text">
            select metrics ({selectedMetrics.length}/{availableMetrics.length})
          </span>
          <ChevronDown size={12} class="text-ctp-subtext0 md:w-4 md:h-4" />
        </summary>

        <div
          class="absolute top-full left-0 right-0 mt-1 bg-ctp-base border border-ctp-surface0/30 shadow-lg z-10 max-h-60 overflow-y-auto"
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
      {#if isLoading}
        <div
          class="absolute inset-0 flex items-center justify-center bg-ctp-base/80 backdrop-blur-sm"
        >
          <div class="animate-pulse text-ctp-blue text-xs">loading data...</div>
        </div>
      {/if}
      <div class="absolute inset-0">
        <canvas bind:this={chartCanvas} class="chart-canvas"></canvas>
      </div>
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
