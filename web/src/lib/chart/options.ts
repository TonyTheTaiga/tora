import type { EChartsCoreOption, SeriesOption } from "echarts/core";
import type { ChartTheme } from "./theme";

export type YScale = "log" | "linear";

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
      axisPointer: { type: "line" },
      valueFormatter: (v: number | string) =>
        typeof v === "number" && Number.isFinite(v)
          ? v.toFixed(4)
          : String(v ?? ""),
      backgroundColor: (theme.terminalBg || theme.mantle) + "ee",
      borderColor: theme.terminalBorder || theme.overlay0 + "44",
      textStyle: { color: theme.text },
    },
    legend: { top: 0, textStyle: { color: theme.text } },
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
    emphasis: { focus: "series" },
  })) as unknown as SeriesOption[];
}
