<script lang="ts">
  import type { Experiment } from "$lib/types";
  import Chart from "chart.js/auto";
  import { ChartLine, ChevronDown } from "lucide-svelte";
  import { onMount, onDestroy } from "svelte";
  import { startTimer } from "$lib/utils/timing";

  let chartInstance: Chart | null = null;
  let chartCanvas: HTMLCanvasElement | null = $state(null);
  interface ExperimentWithMetricData extends Experiment {
    metricData?: Record<string, number[]>;
  }

  let {
    experiment,
    selectedMetrics = $bindable([]),
  }: {
    experiment: ExperimentWithMetricData;
    selectedMetrics?: string[];
  } = $props();
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

  function tooltipReset(node: HTMLCanvasElement) {
    const clear = () => {
      if (chartInstance && chartInstance.tooltip) {
        if (typeof chartInstance.tooltip.setActiveElements === "function") {
          chartInstance.tooltip.setActiveElements([], { x: 0, y: 0 });
          chartInstance.update("none");
        }
      }
    };

    node.addEventListener("touchend", clear);
    node.addEventListener("touchcancel", clear);
    node.addEventListener("pointerup", clear);
    node.addEventListener("pointercancel", clear);

    return {
      destroy() {
        node.removeEventListener("touchend", clear);
        node.removeEventListener("touchcancel", clear);
        node.removeEventListener("pointerup", clear);
        node.removeEventListener("pointercancel", clear);
      },
    };
  }

  onDestroy(() => {
    destroyChart();
  });

  function downloadChart() {
    if (!chartCanvas) return;
    const link = document.createElement("a");
    link.href = chartCanvas.toDataURL("image/png");
    link.download = `${experiment.id}-chart.png`;
    link.click();
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

    if (!chartCanvas || selectedMetrics.length === 0) {
      destroyChart();
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

        return {
          label: metric,
          data: rawDataPoints,
          borderColor: color.border,
          backgroundColor: color.border + "20",
          fill: true,
          pointBackgroundColor: color.point,
          pointBorderColor: ui.base,
          pointHoverBackgroundColor: getComputedStyle(document.documentElement)
            .getPropertyValue("--color-ctp-mauve")
            .trim(),
          pointHoverBorderColor: ui.base,
          borderWidth: 2,
          tension: 0.25,
          pointRadius: 2,
          pointHoverRadius: 8,
        };
      });

      if (!chartInstance) {
        chartInstance = new Chart(chartCanvas, {
          type: "line",
          data: { datasets },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
          parsing: false,
          normalized: true,
          events: [
            "mousemove",
            "mouseout",
            "click",
            "touchstart",
            "touchmove",
            "touchend",
            "touchcancel",
            "mouseup",
            "pointerup",
          ],
          interaction: {
            mode: "nearest",
            intersect: false,
            axis: "x",
          },
          elements: {
            line: {
              borderCapStyle: "round",
              borderJoinStyle: "round",
              borderWidth: 2,
            },
            point: {
              radius: 2,
              hoverRadius: 8,
            },
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
            decimation: {
              enabled: true,
              algorithm: "lttb",
              samples: 50,
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
                color: ui.axisTicks,
                maxTicksLimit: 6,
                autoSkip: true,
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
                color: ui.axisTicks,
                maxTicksLimit: 6,
                autoSkip: true,
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

              if (
                event.type === "mouseout" ||
                eventType === "touchend" ||
                eventType === "touchcancel" ||
                eventType === "mouseup" ||
                eventType === "pointerup"
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
              if (eventType === "mouseleave" || eventType === "pointerleave") {
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
      } else {
        chartInstance.data.datasets = datasets;
        chartInstance.update();
        timer.end({ success: "true" });
        return;
      }
      timer.end({ success: "true" });
    } catch (error) {
      timer.end({
        error: error instanceof Error ? error.message : "Unknown error",
      });
      console.error("Failed to create chart:", error);
    }
  }

  function loadMetricData(metric: string) {
    if (!metricsData[metric]) {
      const values = experiment.metricData?.[metric] || [];
      const steps = values.map((_, i) => i);
      metricsData[metric] = { steps, values };
    }
  }

  $effect(() => {
    if (selectedMetrics.length > 0) {
      for (const metric of selectedMetrics) {
        loadMetricData(metric);
      }
      updateChart();
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
      <div class="absolute inset-0">
        <canvas bind:this={chartCanvas} class="chart-canvas" use:tooltipReset></canvas>
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

