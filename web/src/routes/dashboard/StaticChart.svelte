<script lang="ts">
  import * as echarts from "echarts/core";
  import type { EChartsType } from "echarts/core";
  import { LineChart } from "echarts/charts";
  import {
    GridComponent,
    TooltipComponent,
    LegendComponent,
    DataZoomComponent,
  } from "echarts/components";
  import { CanvasRenderer } from "echarts/renderers";
  import { onMount } from "svelte";
  import { getChartTheme } from "$lib/chart/theme";
  import {
    baseOptions,
    themeAxisUpdate,
    transformForScale,
    lineSeriesFrom,
  } from "$lib/chart/options";

  echarts.use([
    LineChart,
    GridComponent,
    TooltipComponent,
    LegendComponent,
    DataZoomComponent,
    CanvasRenderer,
  ]);

  type LogRow = {
    name: string;
    value: number;
    step: number;
    metadata?: Record<string, any> | null;
    created_at?: string;
  };

  let {
    experimentId,
    yScale,
    refreshKey = 0,
  } = $props<{
    experimentId: string;
    yScale: "log" | "linear";
    refreshKey?: number;
  }>();
  let chartEl: HTMLDivElement | null = null;
  let chart: EChartsType | null = null;
  let ro: ResizeObserver | null = null;
  let loading = $state(false);
  let error: string | null = $state(null);
  let seriesRaw: Record<string, Array<[number, number]>> = $state({});
  let seriesNames = $derived(Object.keys(seriesRaw || {}));
  let hasMetrics = $derived(seriesNames.length > 0);
  let chartTheme = $state(getChartTheme());
  let ac: AbortController | null = null;
  let lastRefreshKey = $state<number | null>(null);

  $effect(() => {
    void yScale;
    void chartTheme;
    if (chart) applyTheme();
  });

  function initChart() {
    if (chart || !chartEl) return;
    chart = echarts.init(chartEl, undefined, { renderer: "canvas" });
    chart.setOption(baseOptions(chartTheme, yScale), { notMerge: true });
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

  function applyTheme() {
    const byScale = transformForScale(seriesRaw, yScale);
    const updates = seriesNames.map((n) => ({ id: n, data: byScale[n] }));
    chart?.setOption(
      { ...themeAxisUpdate(chartTheme, yScale), series: updates },
      { notMerge: false },
    );
  }

  function updateSeries() {
    if (!chart) return;
    const names = seriesNames;
    if (names.length === 0) return;
    const byScale = transformForScale(seriesRaw, yScale);
    const series = lineSeriesFrom(byScale);
    chart.setOption(
      {
        series,
        legend: { data: names },
      },
      { notMerge: false },
    );
  }

  function loadStaticData(controller: AbortController) {
    if (!experimentId) return;
    loading = true;
    error = null;
    (async () => {
      try {
        const resp = await fetch(`/api/experiments/${experimentId}/metrics`, {
          signal: controller.signal,
          cache: "no-store",
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
        if (ac === controller) {
          seriesRaw = byNameRaw;
          initChart();
          updateSeries();
        }
      } catch (e) {
        if (ac === controller) {
          error = e instanceof Error ? e.message : "Failed to load chart data";
          seriesRaw = {};
        }
      } finally {
        if (ac === controller) {
          loading = false;
        }
      }
    })();
  }

  onMount(() => {
    initChart();
    if (chartEl && typeof ResizeObserver !== "undefined") {
      ro = new ResizeObserver(() => chart?.resize());
      ro.observe(chartEl);
    }
    const handleThemeChange = () => {
      chartTheme = getChartTheme();
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

    ac = new AbortController();
    loadStaticData(ac);

    return async () => {
      if (mediaQuery)
        mediaQuery.removeEventListener("change", handleThemeChange);
      if (observer) observer.disconnect();
      if (ro) {
        try {
          ro.disconnect();
        } catch {}
        ro = null;
      }
      try {
        ac?.abort();
      } catch {}
      ac = null;
      await Promise.resolve();
      disposeChart();
    };
  });

  $effect(() => {
    void refreshKey;
    if (lastRefreshKey === null) {
      lastRefreshKey = refreshKey;
      return;
    }
    if (refreshKey === lastRefreshKey) return;
    lastRefreshKey = refreshKey;
    try {
      ac?.abort();
    } catch {}
    ac = new AbortController();
    loadStaticData(ac);
  });
</script>

<div
  class="relative h-80 w-full bg-transparent border border-ctp-surface0/20 overflow-hidden"
>
  <div class="absolute inset-0 py-2" bind:this={chartEl}></div>
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
