import Chart from "chart.js/auto";
import type { ExperimentWithMetrics } from "./+page.server";

export function drawScatterChart(
  canvasElement: HTMLCanvasElement,
  experiments: ExperimentWithMetrics[],
  selectedMetrics: [string, string],
  experimentColors: Map<string, string>,
): Chart | null {
  const computedStyles = getComputedStyle(document.documentElement);
  const [xMetric, yMetric] = selectedMetrics;

  const datasets = experiments.map((experiment) => {
    const xValues = experiment.metricData?.[xMetric];
    const yValues = experiment.metricData?.[yMetric];

    const xValue = xValues?.length ? xValues[xValues.length - 1] : 0;
    const yValue = yValues?.length ? yValues[yValues.length - 1] : 0;

    const color = experimentColors.get(experiment.id) || "#666";

    return {
      label: experiment.name,
      data: [{ x: xValue, y: yValue }],
      backgroundColor: `${color}80`,
      borderColor: color,
      borderWidth: 2,
      pointRadius: 6,
      pointHoverRadius: 10,
    };
  });

  const chart = new Chart(canvasElement, {
    type: "scatter",
    data: {
      datasets: datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
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
      plugins: {
        legend: {
          display: false,
        },
        tooltip: {
          enabled: true,
          mode: "nearest",
          intersect: false,
          callbacks: {
            title: function (context) {
              return context[0].dataset.label || "Experiment";
            },
            label: function (context) {
              const point = context.parsed;
              return [
                `${xMetric}: ${point.x.toFixed(3)}`,
                `${yMetric}: ${point.y.toFixed(3)}`,
              ];
            },
          },
        },
      },
      scales: {
        x: {
          type: "linear",
          position: "bottom",
          grid: {
            color: `${computedStyles.getPropertyValue("--color-ctp-overlay0").trim()}30`,
            lineWidth: 1,
          },
          ticks: {
            color:
              computedStyles.getPropertyValue("--color-ctp-subtext0").trim() ||
              "#666",
            maxTicksLimit: 6,
            autoSkip: true,
          },
          title: {
            display: true,
            text: xMetric,
            color:
              computedStyles.getPropertyValue("--color-ctp-subtext1").trim() ||
              "#888",
          },
        },
        y: {
          type: "linear",
          grid: {
            color: `${computedStyles.getPropertyValue("--color-ctp-overlay0").trim()}30`,
            lineWidth: 1,
          },
          ticks: {
            color:
              computedStyles.getPropertyValue("--color-ctp-subtext0").trim() ||
              "#666",
            maxTicksLimit: 6,
            autoSkip: true,
          },
          title: {
            display: true,
            text: yMetric,
            color:
              computedStyles.getPropertyValue("--color-ctp-subtext1").trim() ||
              "#888",
          },
        },
      },
      interaction: {
        intersect: false,
        mode: "nearest",
      },
      elements: {
        point: {
          radius: 8,
          hoverRadius: 12,
          borderWidth: 2,
          hoverBorderWidth: 3,
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



          // Clear tooltips when interaction ends
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

  return chart;
}
