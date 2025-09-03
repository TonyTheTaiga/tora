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
  import type { Snippet } from "svelte";

  echarts.use([
    LineChart,
    GridComponent,
    TooltipComponent,
    LegendComponent,
    DataZoomComponent,
    CanvasRenderer,
  ]);

  let { experimentId, toggleStreaming } = $props<{
    experimentId: string;
    toggleStreaming?: Snippet;
  }>();
  let yScale = $state<"log" | "linear">("log");
  function toggleScale() {
    yScale = yScale === "log" ? "linear" : "log";
    // apply immediately on toggle to avoid extra effects
    applyTheme();
  }

  let chartEl: HTMLDivElement | null = $state(null);
  let chart: EChartsType | null = null;
  let loading = $state(false);
  let error: string | null = $state(null);
  let hasMetrics = $state(false);
  let seriesRaw: Record<string, Array<[number, number]>> = $state({});
  let seriesNames: string[] = $state([]);

  type LogRow = {
    name: string;
    value: number;
    step: number;
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

  async function loadStaticData() {
    if (!experimentId) return;
    loading = true;
    error = null;
    hasMetrics = false;
    try {
      const resp = await fetch(`/api/experiments/${experimentId}/metrics`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const json = await resp.json();
      const metrics = (json?.data ?? []) as LogRow[];
      // Group by name into raw numeric series (preserve API order)
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

      const names = Object.keys(byNameRaw);
      seriesRaw = byNameRaw;
      seriesNames = names;
      hasMetrics = names.length > 0;
      const byScale = dataForScale(seriesRaw);
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
    const byScale = dataForScale(seriesRaw);
    const updates = seriesNames.map((n) => ({ id: n, data: byScale[n] }));
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
    return () => {
      if (mediaQuery)
        mediaQuery.removeEventListener("change", handleThemeChange);
      if (observer) observer.disconnect();
      disposeChart();
    };
  });

  // Fetch when experiment changes
  $effect(() => {
    const id = experimentId;
    if (!id) return;
    loadStaticData();
  });
</script>

<div
  class="relative h-80 w-full border border-ctp-surface0/20 bg-transparent overflow-hidden"
>
  <div class="absolute inset-0" bind:this={chartEl}></div>
  <div class="absolute top-1 right-1 flex gap-1 z-10">
    <button
      class="text-[10px] leading-none border border-ctp-surface0/40 px-1.5 py-0.5 bg-ctp-mantle/70 hover:text-ctp-blue"
      onclick={toggleScale}
      title="toggle Y axis scale between log and linear"
    >
      y: {yScale}
    </button>
    {@render toggleStreaming?.()}
  </div>
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
