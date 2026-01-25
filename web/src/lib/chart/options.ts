import type { EChartsCoreOption } from "echarts/core";
import type { SeriesOption } from "echarts/types/dist/shared";
import type { ChartTheme } from "./theme";

export type YScale = "log" | "linear";

function formatValue(value: number): string {
  if (!Number.isFinite(value)) return "N/A";
  if (Math.abs(value) >= 1000000) return (value / 1000000).toFixed(2) + "M";
  if (Math.abs(value) >= 1000) return (value / 1000).toFixed(2) + "K";
  if (Math.abs(value) < 0.01 && value !== 0) return value.toExponential(2);
  return value.toFixed(4);
}

export function baseOptions(
  theme: ChartTheme,
  yScale: YScale = "log",
): EChartsCoreOption {
  return {
    animation: true,
    color: theme.colors as any,
    textStyle: { color: theme.text },
    grid: { left: 56, right: 20, top: 24, bottom: 40 },
    tooltip: {
      trigger: "axis",
      axisPointer: {
        type: "cross",
        crossStyle: {
          color: theme.overlay0,
        },
        lineStyle: {
          color: theme.overlay0,
          type: "dashed",
        },
        label: {
          backgroundColor: theme.mantle,
          color: theme.text,
          borderColor: theme.overlay0,
        },
      },
      backgroundColor: (theme.terminalBg || theme.mantle) + "f5",
      borderColor: theme.terminalBorder || theme.overlay0 + "44",
      borderWidth: 1,
      padding: [8, 12],
      textStyle: { color: theme.text, fontSize: 12 },
      formatter: (params: any) => {
        if (!Array.isArray(params) || params.length === 0) return "";

        const step = params[0].data[0];
        let html = `<div style="font-weight: 600; margin-bottom: 6px; border-bottom: 1px solid ${theme.overlay0}40; padding-bottom: 4px;">Step: ${step}</div>`;

        for (const p of params) {
          const value = p.data[1];
          const formattedValue =
            typeof value === "number" && Number.isFinite(value)
              ? formatValue(value)
              : String(value ?? "N/A");

          html += `<div style="display: flex; align-items: center; gap: 8px; margin: 4px 0;">
            <span style="display: inline-block; width: 10px; height: 10px; background: ${p.color}; border-radius: 2px;"></span>
            <span style="flex: 1; color: ${theme.text}80;">${p.seriesName}</span>
            <span style="font-family: monospace; font-weight: 500;">${formattedValue}</span>
          </div>`;
        }

        return html;
      },
    },
    legend: {
      top: 0,
      textStyle: { color: theme.text },
      selectedMode: "multiple",
      inactiveColor: theme.overlay0,
    },
    dataZoom: [{ type: "inside", xAxisIndex: 0 }],
    xAxis: {
      type: "value",
      name: "step",
      nameGap: 14,
      boundaryGap: [0, 0],
      axisLabel: { color: theme.axisTicks },
      axisLine: { lineStyle: { color: theme.overlay0 } },
      splitLine: { show: true, lineStyle: { color: theme.fadedGridLines } },
    },
    yAxis: {
      type: yScale === "log" ? "log" : "value",
      name: "value",
      minorTick: { show: true },
      min: "dataMin",
      max: "dataMax",
      scale: true,
      axisLabel: {
        color: theme.axisTicks,
        width: 56,
        overflow: "truncate",
        align: "right",
        formatter: (value: number) => {
          if (Math.abs(value) >= 1000000)
            return (value / 1000000).toFixed(1) + "M";
          if (Math.abs(value) >= 1000) return (value / 1000).toFixed(1) + "K";
          if (Math.abs(value) < 0.01 && value !== 0)
            return value.toExponential(1);
          return value.toFixed(2);
        },
      },
      axisLine: { lineStyle: { color: theme.overlay0 } },
      splitLine: { show: true, lineStyle: { color: theme.fadedGridLines } },
    },
    series: [],
  } satisfies EChartsCoreOption;
}

export function themeAxisUpdate(theme: ChartTheme, yScale: YScale) {
  return {
    color: theme.colors as any,
    textStyle: { color: theme.text },
    legend: { textStyle: { color: theme.text } },
    tooltip: {
      backgroundColor: (theme.terminalBg || theme.mantle) + "ee",
      borderColor: theme.terminalBorder || theme.overlay0 + "44",
      textStyle: { color: theme.text },
    },
    xAxis: [
      {
        axisLabel: { color: theme.axisTicks },
        axisLine: { lineStyle: { color: theme.overlay0 } },
        splitLine: { show: true, lineStyle: { color: theme.fadedGridLines } },
      },
    ],
    yAxis: [
      {
        type: yScale === "log" ? "log" : "value",
        min: "dataMin",
        max: "dataMax",
        scale: true,
        axisLabel: {
          color: theme.axisTicks,
          width: 56,
          overflow: "truncate",
          align: "right",
        },
        axisLine: { lineStyle: { color: theme.overlay0 } },
        splitLine: { show: true, lineStyle: { color: theme.fadedGridLines } },
      },
    ],
  } as Partial<EChartsCoreOption>;
}

export type SeriesData = Record<string, Array<[number, number]>>;
export type SeriesDataScaled = Record<string, Array<[number, number | null]>>;

export function transformForScale(
  raw: SeriesData,
  yScale: YScale,
): SeriesDataScaled {
  const out: SeriesDataScaled = {};
  for (const n of Object.keys(raw)) {
    const arr = raw[n];
    out[n] = arr.map(([x, y]) =>
      yScale === "log"
        ? [x, y > 0 ? y : null]
        : [x, Number.isFinite(y) ? y : null],
    );
  }
  return out;
}

export function lineSeriesFrom(byScale: SeriesDataScaled): SeriesOption[] {
  return Object.keys(byScale).map((n) => ({
    id: n,
    name: n,
    type: "line",
    showSymbol: false,
    smooth: 0.15,
    connectNulls: true,
    data: byScale[n],
    symbolSize: 6,
    emphasis: {
      focus: "series",
      lineStyle: {
        width: 3,
      },
    },
  })) as unknown as SeriesOption[];
}
