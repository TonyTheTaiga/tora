import Chart from "chart.js/auto";
import type { ExperimentWithMetrics } from "./+page.server";

export function drawRadarChart(
  canvasElement: HTMLCanvasElement,
  experiments: ExperimentWithMetrics[],
  selectedMetrics: string[],
  experimentColors: Map<string, string>,
): Chart | null {
  const computedStyles = getComputedStyle(document.documentElement);
  const surfaceColor = computedStyles
    .getPropertyValue("--color-ctp-surface0")
    .trim();

  const datasets = experiments.map((experiment) => {
    const experimentColor = experimentColors.get(experiment.id);

    const dataPoints = selectedMetrics.map((metricName) => {
      const metricValues = experiment.metricData?.[metricName];
      if (
        !metricValues ||
        !Array.isArray(metricValues) ||
        metricValues.length === 0
      ) {
        return 0;
      }
      return metricValues[metricValues.length - 1];
    });

    return {
      label: experiment.name,
      data: dataPoints,
      fill: false,
      backgroundColor: `${experimentColor}30`,
      borderColor: experimentColor,
      pointBackgroundColor: experimentColor,
      pointBorderColor: surfaceColor || "#fff",
      pointHoverBackgroundColor: surfaceColor || "#fff",
      pointHoverBorderColor: experimentColor,
    };
  });

  const chart = new Chart(canvasElement, {
    type: "radar",
    data: {
      labels: selectedMetrics,
      datasets: datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      aspectRatio: 1,
      events: ["mousemove", "mouseout", "click", "touchstart", "touchmove"],
      elements: {
        line: {
          borderWidth: 3,
          tension: 0.1,
        },
        point: {
          radius: 6,
          hoverRadius: 10,
          borderWidth: 2,
          hoverBorderWidth: 3,
        },
      },
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
              const metricName = selectedMetrics[context.dataIndex];
              const value = context.parsed.r;
              return `${metricName}: ${value.toFixed(3)}`;
            },
          },
        },
      },
      scales: {
        r: {
          beginAtZero: true,
          grid: {
            color: `${computedStyles.getPropertyValue("--color-ctp-overlay0").trim()}30`,
            lineWidth: 1,
          },
          angleLines: {
            color: `${computedStyles.getPropertyValue("--color-ctp-overlay0").trim()}40`,
            lineWidth: 1,
          },
          pointLabels: {
            color:
              computedStyles.getPropertyValue("--color-ctp-subtext1").trim() ||
              "#888",
            font: {
              size: 12,
            },
          },
          ticks: {
            color:
              computedStyles.getPropertyValue("--color-ctp-subtext0").trim() ||
              "#666",
            backdropColor: "transparent",
            stepSize: 0.1,
            callback: function (value: string | number) {
              return typeof value === "number" ? value.toFixed(1) : value;
            },
          },
        },
      },
      interaction: {
        intersect: false,
        mode: "nearest",
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

          // Prevent scrolling on touch events
          if (eventType === "touchstart" || eventType === "touchmove") {
            if (event.native) {
              event.native.preventDefault();
            }
          }

          // Clear tooltips when interaction ends
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

  return chart;
}
