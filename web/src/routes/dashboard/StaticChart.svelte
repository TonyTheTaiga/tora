<script lang="ts">
  import * as echarts from "echarts/core";
  import type { EChartsType, EChartsCoreOption } from "echarts/core";
  import { LineChart } from "echarts/charts";
  import {
    GridComponent,
    TooltipComponent,
    LegendComponent,
    DataZoomComponent,
  } from "echarts/components";
  import { CanvasRenderer } from "echarts/renderers";
  import { browser } from "$app/environment";
  // icons removed from internal overlay; toolbar handles controls

  echarts.use([
    LineChart,
    GridComponent,
    TooltipComponent,
    LegendComponent,
    DataZoomComponent,
    CanvasRenderer,
  ]);

  let { experimentId } = $props<{
    experimentId: string;
  }>();
  let yScale = $state<"log" | "linear">("log");
  export function toggleScale() {
    yScale = yScale === "log" ? "linear" : "log";
    applyTheme();
  }

  let chartEl: HTMLDivElement | null = $state(null);
  let chart: EChartsType | null = null;
  let loading = $state(false);
  let error: string | null = $state(null);
  let seriesRaw: Record<string, Array<[number, number]>> = $state({});
  let seriesNames = $derived(Object.keys(seriesRaw || {}));
  let hasMetrics = $derived(seriesNames.length > 0);

  type LogRow = {
    name: string;
    value: number;
    step: number;
    metadata?: Record<string, any> | null;
    created_at?: string;
  };

  const CHART_COLOR_KEYS = [
    "blue",
    "lavender",
    "sky",
    "green",
    "teal",
    "mauve",
    "peach",
    "yellow",
    "pink",
    "sapphire",
    "maroon",
    "red",
    "rosewater",
  ];

  function getTheme() {
    if (!browser) {
      return {
        colors: ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"],
        text: "#111",
        mantle: "#fff",
        overlay0: "#888",
        sky: "#2aa1ff",
        fadedGridLines: "#ddd",
        axisTicks: "#666",
      } as const;
    }
    const cs = getComputedStyle(document.documentElement);
    const colors = CHART_COLOR_KEYS.map((k) =>
      cs.getPropertyValue(`--color-ctp-${k}`).trim(),
    ).filter(Boolean);
    return {
      colors: colors.length ? colors : ["#4e79a7", "#f28e2b", "#e15759"],
      text: cs.getPropertyValue("--color-ctp-text").trim(),
      mantle: cs.getPropertyValue("--color-ctp-mantle").trim(),
      overlay0: cs.getPropertyValue("--color-ctp-overlay0").trim(),
      sky: cs.getPropertyValue("--color-ctp-sky").trim(),
      fadedGridLines: cs.getPropertyValue("--color-ctp-surface1").trim() + "33",
      axisTicks: cs.getPropertyValue("--color-ctp-subtext0").trim(),
      terminalBg: cs.getPropertyValue("--color-ctp-terminal-bg").trim(),
      terminalBorder: cs.getPropertyValue("--color-ctp-terminal-border").trim(),
    } as const;
  }

  let chartTheme = $state(getTheme());

  function initChart() {
    if (chart || !chartEl) return;
    chart = echarts.init(chartEl, undefined, { renderer: "canvas" });
    chart.setOption(getBaseOptions(), { notMerge: true });
    window.addEventListener("resize", handleResize);
  }

  function disposeChart() {
    if (chart) {
      chart.dispose();
      chart = null;
    }
    window.removeEventListener("resize", handleResize);
  }

  function handleResize() {
    chart?.resize();
  }

  function getBaseOptions(): EChartsCoreOption {
    return {
      animation: true,
      color: chartTheme.colors as any,
      textStyle: { color: chartTheme.text },
      grid: { left: 40, right: 20, top: 24, bottom: 40 },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "line" },
        valueFormatter: (v: number | string) =>
          typeof v === "number" && Number.isFinite(v)
            ? v.toFixed(4)
            : String(v ?? ""),
        backgroundColor: (chartTheme.terminalBg || chartTheme.mantle) + "ee",
        borderColor: chartTheme.terminalBorder || chartTheme.overlay0 + "44",
        textStyle: { color: chartTheme.text },
      },
      legend: { top: 0, textStyle: { color: chartTheme.text } },
      dataZoom: [{ type: "inside", xAxisIndex: 0 }],
      xAxis: {
        type: "value",
        name: "step",
        nameGap: 14,
        boundaryGap: [0, 0],
        axisLabel: { color: chartTheme.axisTicks },
        axisLine: { lineStyle: { color: chartTheme.overlay0 } },
        splitLine: {
          show: true,
          lineStyle: { color: chartTheme.fadedGridLines },
        },
      },
      yAxis: {
        type: yScale === "log" ? "log" : "value",
        name: "value",
        minorTick: { show: true },
        min: "dataMin",
        max: "dataMax",
        scale: true,
        axisLabel: { color: chartTheme.axisTicks },
        axisLine: { lineStyle: { color: chartTheme.overlay0 } },
        splitLine: {
          show: true,
          lineStyle: { color: chartTheme.fadedGridLines },
        },
      },
      series: [],
    };
  }

  function dataForScale(raw: Record<string, Array<[number, number]>>) {
    const out: Record<string, Array<[number, number | null]>> = {};
    for (const n of Object.keys(raw)) {
      const arr = raw[n];
      if (yScale === "log") {
        out[n] = arr.map(([x, y]) => [x, y > 0 ? y : null]);
      } else {
        out[n] = arr.map(([x, y]) => [x, Number.isFinite(y) ? y : null]);
      }
    }
    return out;
  }

  function applyTheme() {
    const byScale = dataForScale(seriesRaw);
    const updates = seriesNames.map((n) => ({ id: n, data: byScale[n] }));
    chart?.setOption(
      {
        color: chartTheme.colors as any,
        textStyle: { color: chartTheme.text },
        legend: { textStyle: { color: chartTheme.text } },
        tooltip: {
          backgroundColor: (chartTheme.terminalBg || chartTheme.mantle) + "ee",
          borderColor: chartTheme.terminalBorder || chartTheme.overlay0 + "44",
          textStyle: { color: chartTheme.text },
        },
        series: updates,
        xAxis: [
          {
            axisLabel: { color: chartTheme.axisTicks },
            axisLine: { lineStyle: { color: chartTheme.overlay0 } },
            splitLine: {
              show: true,
              lineStyle: { color: chartTheme.fadedGridLines },
            },
          },
        ],
        yAxis: [
          {
            type: yScale === "log" ? "log" : "value",
            min: "dataMin",
            max: "dataMax",
            scale: true,
            axisLabel: { color: chartTheme.axisTicks },
            axisLine: { lineStyle: { color: chartTheme.overlay0 } },
            splitLine: {
              show: true,
              lineStyle: { color: chartTheme.fadedGridLines },
            },
          },
        ],
      },
      { notMerge: false },
    );
  }
  import { onMount } from "svelte";
  let metricsAbort: AbortController | null = null;
  function loadStaticData(controller: AbortController) {
    if (!experimentId) return;
    loading = true;
    error = null;
    (async () => {
      try {
        const resp = await fetch(`/api/experiments/${experimentId}/metrics`, {
          signal: controller.signal,
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const json = await resp.json();
        const metrics = (json?.data ?? []) as LogRow[];
        const byNameRaw: Record<string, Array<[number, number]>> = {};
        for (const r of metrics) {
          const name = String(r.name);
          const y =
            typeof r.value === "number" && Number.isFinite(r.value)
              ? r.value
              : NaN;
          const step = r.step as number;
          if (!byNameRaw[name]) byNameRaw[name] = [];
          byNameRaw[name].push([step, y]);
        }
        if (metricsAbort === controller) {
          seriesRaw = byNameRaw;
        }
      } catch (e) {
        if (metricsAbort === controller) {
          error = e instanceof Error ? e.message : "Failed to load chart data";
          seriesRaw = {};
        }
      } finally {
        if (metricsAbort === controller) {
          loading = false;
        }
      }
    })();
  }

  export function refreshChart() {
    try {
      metricsAbort?.abort();
    } catch {}
    metricsAbort = new AbortController();
    loadStaticData(metricsAbort);
  }

  onMount(() => {
    initChart();
    const handleThemeChange = () => {
      chartTheme = getTheme();
      applyTheme();
    };
    let mediaQuery: MediaQueryList | null = null;
    let observer: MutationObserver | null = null;
    if (typeof window !== "undefined") {
      mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
      mediaQuery.addEventListener("change", handleThemeChange);
      observer = new MutationObserver((mutations) => {
        for (const m of mutations) {
          if (
            m.attributeName === "class" &&
            m.target === document.documentElement
          ) {
            handleThemeChange();
          }
        }
      });
      observer.observe(document.documentElement, {
        attributes: true,
        attributeFilter: ["class"],
      });
    }
    // Fetch metrics on mount only
    metricsAbort = new AbortController();
    loadStaticData(metricsAbort);

    return () => {
      if (mediaQuery)
        mediaQuery.removeEventListener("change", handleThemeChange);
      if (observer) observer.disconnect();
      try {
        metricsAbort?.abort();
      } catch {}
      metricsAbort = null;
      disposeChart();
    };
  });

  // Update the chart when data or scale changes
  $effect(() => {
    const names = seriesNames;
    const raw = seriesRaw;
    const scale = yScale; // track scale changes
    if (!names || names.length === 0) return;
    initChart();
    const byScale = dataForScale(raw);
    const series = names.map((n) => ({
      id: n,
      name: n,
      type: "line",
      showSymbol: false,
      smooth: 0.15,
      connectNulls: true,
      data: byScale[n],
      emphasis: { focus: "series" },
    }));
    chart?.setOption(
      {
        series,
        legend: { data: names },
      },
      { notMerge: false },
    );
  });

  // Fetching is handled on mount; parent remounts this component on id change
</script>

<div
  class="relative h-80 w-full bg-transparent border border-ctp-surface0/20 overflow-hidden"
>
  <div class="absolute inset-0" bind:this={chartEl}></div>
  {#if loading}
    <div class="absolute inset-0 flex items-center justify-center">
      <div class="text-ctp-subtext0 text-xs">loading chart…</div>
    </div>
  {:else if error}
    <div class="absolute inset-0 flex items-center justify-center">
      <div class="text-ctp-red text-xs">{error}</div>
    </div>
  {:else if !hasMetrics}
    <div class="absolute inset-0 flex items-center justify-center">
      <div class="text-ctp-subtext0 text-xs">no metrics yet</div>
    </div>
  {/if}
</div>
