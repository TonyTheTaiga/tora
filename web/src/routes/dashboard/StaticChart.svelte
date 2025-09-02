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

  echarts.use([
    LineChart,
    GridComponent,
    TooltipComponent,
    LegendComponent,
    DataZoomComponent,
    CanvasRenderer,
  ]);

  let { experimentId } = $props<{ experimentId: string }>();

  let chartEl: HTMLDivElement | null = $state(null);
  let chart: EChartsType | null = null;
  let loading = $state(false);
  let error: string | null = $state(null);
  let hasMetrics = $state(false);

  type LogRow = {
    name: string;
    value: number;
    step?: number | null;
    metadata?: Record<string, any> | null;
    created_at?: string;
  };

  const CHART_COLOR_KEYS = [
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
        backgroundColor: chartTheme.mantle + "cc",
        borderColor: chartTheme.overlay0 + "33",
        textStyle: { color: chartTheme.text },
      },
      legend: { top: 0, textStyle: { color: chartTheme.text } },
      dataZoom: [
        { type: "inside", xAxisIndex: 0 },
        {
          type: "slider",
          xAxisIndex: 0,
          height: 18,
          bottom: 8,
          textStyle: { color: chartTheme.axisTicks },
          borderColor: chartTheme.overlay0 + "33",
          backgroundColor: chartTheme.mantle + "22",
          fillerColor: chartTheme.sky + "33",
          handleStyle: { color: chartTheme.sky },
          dataBackground: {
            lineStyle: { color: chartTheme.overlay0 + "55" },
            areaStyle: { color: chartTheme.overlay0 + "22" },
          },
        },
      ],
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
        type: "log",
        name: "value",
        minorTick: { show: true },
        min: 1e-9,
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

  async function loadStaticData() {
    if (!experimentId) return;
    loading = true;
    error = null;
    hasMetrics = false;
    try {
      const resp = await fetch(`/api/experiments/${experimentId}/logs`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const json = await resp.json();
      const rows = (json?.data ?? []) as LogRow[];
      // Filter only metrics
      const metrics = rows.filter(
        (r) => (r?.metadata as any)?.type === "metric",
      );
      // The API is DESC created_at; reverse to ASC
      metrics.reverse();
      // Group by name
      const byName: Record<string, Array<[number, number]>> = {};
      const counters: Record<string, number> = {};
      for (const r of metrics) {
        const name = String(r.name);
        const y = Number.isFinite(r.value) ? r.value : 0;
        const step =
          typeof r.step === "number" ? r.step : (counters[name] ?? 0);
        counters[name] = (counters[name] ?? 0) + 1;
        if (!byName[name]) byName[name] = [];
        byName[name].push([step, y > 0 ? y : 1e-9]);
      }

      const names = Object.keys(byName);
      hasMetrics = names.length > 0;
      const series = names.map((n) => ({
        name: n,
        type: "line",
        showSymbol: false,
        smooth: 0.15,
        data: byName[n],
        emphasis: { focus: "series" },
      }));

      initChart();
      chart?.setOption(
        {
          series,
          legend: { data: names },
        },
        { notMerge: false },
      );
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to load chart data";
    } finally {
      loading = false;
    }
  }

  function applyTheme() {
    chart?.setOption(
      {
        color: chartTheme.colors as any,
        textStyle: { color: chartTheme.text },
        legend: { textStyle: { color: chartTheme.text } },
        tooltip: {
          backgroundColor: chartTheme.mantle + "cc",
          borderColor: chartTheme.overlay0 + "33",
          textStyle: { color: chartTheme.text },
        },
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

  $effect(() => {
    initChart();
    loadStaticData();
    // Theme change listeners
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
    return () => {
      if (mediaQuery)
        mediaQuery.removeEventListener("change", handleThemeChange);
      if (observer) observer.disconnect();
      disposeChart();
    };
  });

  // Reload if experimentId changes
  $effect(() => {
    const _id = experimentId;
    loadStaticData();
  });
</script>

<div
  class="relative h-80 w-full border border-ctp-surface0/20 bg-transparent overflow-hidden"
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
