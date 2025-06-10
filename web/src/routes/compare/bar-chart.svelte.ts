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
      events: ['mousemove', 'mouseout', 'click', 'touchstart', 'touchmove'],
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
          },
        },
      },
      interaction: {
        intersect: false,
        mode: "index",
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
  });

  return chart;
}
