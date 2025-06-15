import Chart from "chart.js/auto";
import type { ExperimentWithMetrics } from "./+page.server";

export function drawBarChart(
  canvasElement: HTMLCanvasElement,
  experiments: ExperimentWithMetrics[],
  selectedMetric: string,
  experimentColors: Map<string, string>,
): Chart | null {
  const computedStyles = getComputedStyle(document.documentElement);

  const labels = experiments.map(() => "");
  const dataPoints = experiments.map((experiment) => {
    const metricValues = experiment.metricData?.[selectedMetric];

    if (
      !metricValues ||
      !Array.isArray(metricValues) ||
      metricValues.length === 0
    ) {
      return 0;
    }
    return metricValues[metricValues.length - 1];
  });

  const backgroundColors = experiments.map((exp) => {
    const color = experimentColors.get(exp.id);
    return color ? `${color}30` : "#666630";
  });

  const borderColors = experiments.map((exp) => {
    const color = experimentColors.get(exp.id);
    return color || "#666";
  });

  const chart = new Chart(canvasElement, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: selectedMetric,
          data: dataPoints,
          backgroundColor: backgroundColors,
          borderColor: borderColors,
          borderWidth: 2,
        },
      ],
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
          mode: "index",
          intersect: false,
          callbacks: {
            title: function (context) {
              const index = context[0].dataIndex;
              return experiments[index]?.name || `Experiment ${index + 1}`;
            },
            label: function (context) {
              return `${selectedMetric}: ${context.parsed.y.toFixed(3)}`;
            },
          },
        },
      },
      scales: {
        y: {
          beginAtZero: true,
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
            text: selectedMetric,
            color:
              computedStyles.getPropertyValue("--color-ctp-subtext1").trim() ||
              "#888",
          },
        },
        x: {
          grid: {
            color: `${computedStyles.getPropertyValue("--color-ctp-overlay0").trim()}20`,
            lineWidth: 1,
          },
          ticks: {
            color:
              computedStyles.getPropertyValue("--color-ctp-subtext0").trim() ||
              "#666",
            maxRotation: 45,
            maxTicksLimit: 6,
            autoSkip: true,
          },
        },
      },
      interaction: {
        intersect: false,
        mode: "nearest",
        axis: "x",
      },
      elements: {
        bar: {
          borderRadius: 4,
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
