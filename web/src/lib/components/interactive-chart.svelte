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
  let availableMetrics = $derived(experiment.availableMetrics || []);
  let filteredMetrics = $derived(
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
      gridLines: `${computedStyles.getPropertyValue("--color-ctp-overlay0").trim()}15`, // hex transparency
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
          backgroundColor: color.bg,
          fill: false,
          pointBackgroundColor: color.point,
          pointBorderColor: ui.mantle,
          pointHoverBackgroundColor: getComputedStyle(document.documentElement)
            .getPropertyValue("--color-ctp-mauve")
            .trim(),
          pointHoverBorderColor: ui.base,
          borderWidth: 1.5,
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
          events: ["mousemove", "mouseout", "click", "touchstart", "touchmove"],
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
              backgroundColor: ui.crust,
              titleColor: getComputedStyle(document.documentElement)
                .getPropertyValue("--color-ctp-sky")
                .trim(),
              bodyColor: ui.text,
              borderColor: ui.overlay0,
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

<div class="p-3 sm:p-4 space-y-4 w-full">
  <!-- Metric Selector -->
  {#if availableMetrics.length > 0}
    <div class="border-b border-ctp-surface1 pb-4">
      <details class="relative">
        <summary
          class="flex items-center justify-between cursor-pointer p-2 bg-ctp-surface0 rounded border border-ctp-surface1 hover:bg-ctp-surface1 transition-colors"
        >
          <span class="text-sm text-ctp-text">
            Select metrics ({selectedMetrics.length} of {availableMetrics.length})
          </span>
          <ChevronDown size={16} class="text-ctp-subtext1" />
        </summary>

        <div
          class="absolute top-full left-0 right-0 mt-1 bg-ctp-surface0 border border-ctp-surface1 rounded shadow-lg z-10 max-h-60 overflow-y-auto"
        >
          <!-- Search filter -->
          <div class="p-2 border-b border-ctp-surface1">
            <input
              type="search"
              placeholder="Filter metrics..."
              bind:value={searchFilter}
              class="w-full px-2 py-1 text-sm bg-ctp-base border border-ctp-surface1 rounded text-ctp-text placeholder-ctp-subtext0 focus:outline-none focus:border-ctp-blue"
            />
          </div>

          <!-- Control buttons -->
          <div class="flex gap-2 p-2 border-b border-ctp-surface1">
            <button
              onclick={selectAllMetrics}
              class="px-2 py-1 text-xs bg-ctp-green/20 text-ctp-green rounded hover:bg-ctp-green/30 transition-colors"
            >
              Select All
            </button>
            <button
              onclick={clearAllMetrics}
              class="px-2 py-1 text-xs bg-ctp-red/20 text-ctp-red rounded hover:bg-ctp-red/30 transition-colors"
            >
              Clear All
            </button>
          </div>

          <!-- Metric checkboxes -->
          <div class="p-1">
            {#each filteredMetrics as metric}
              <label
                class="flex items-center gap-2 p-2 hover:bg-ctp-surface1 rounded cursor-pointer"
              >
                <input
                  type="checkbox"
                  checked={selectedMetrics.includes(metric)}
                  onchange={() => toggleMetricCheckbox(metric)}
                  class="text-ctp-blue focus:ring-ctp-blue focus:ring-2"
                />
                <span class="text-sm text-ctp-text">{metric}</span>
              </label>
            {/each}

            {#if filteredMetrics.length === 0}
              <div class="p-2 text-sm text-ctp-subtext0 text-center">
                No metrics found
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
      class="relative h-60 sm:h-80 w-full rounded-md border border-ctp-surface1 bg-ctp-mantle overflow-hidden shadow-md"
    >
      {#if isLoading}
        <div
          class="absolute inset-0 flex items-center justify-center bg-ctp-mantle/80 backdrop-blur-sm"
        >
          <div class="animate-pulse text-[#91d7e3]">Loading data...</div>
        </div>
      {/if}
      <div class="absolute inset-0 p-2 sm:p-4">
        <canvas bind:this={chartCanvas} class="chart-canvas"></canvas>
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
  .chart-canvas {
    background-color: transparent;
    border-radius: 4px;
    touch-action: none;
    user-select: none;
    -webkit-user-select: none;
    -webkit-touch-callout: none;
  }
</style>
