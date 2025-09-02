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
  import { env } from "$env/dynamic/public";

  echarts.use([
    LineChart,
    GridComponent,
    TooltipComponent,
    LegendComponent,
    DataZoomComponent,
    CanvasRenderer,
  ]);

  let {
    experimentId,
    status = $bindable<"idle" | "connecting" | "open" | "closed" | "error">(
      "idle",
    ),
  } = $props<{
    experimentId: string;
    status?: "idle" | "connecting" | "open" | "closed" | "error";
  }>();

  let chartEl: HTMLDivElement | null = $state(null);
  let chart: EChartsType | null = null;
  let seriesData: Record<string, Array<[number, number]>> = {};
  let pending: Record<string, Array<[number, number]>> = {};
  let updateScheduled = false;

  let ws: WebSocket | null = null;
  let reconnectDelayMs = 500;
  let closing = false;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

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

  function ensureSeries(name: string) {
    if (!seriesData[name]) {
      seriesData[name] = [];
      const names = Object.keys(seriesData);
      const newSeries = names.map((n) => ({
        name: n,
        type: "line",
        showSymbol: false,
        smooth: 0.15,
        data: seriesData[n],
        emphasis: { focus: "series" },
      }));
      chart?.setOption(
        { series: newSeries, legend: { data: names } },
        { notMerge: false },
      );
    }
  }

  function enqueue(name: string, step: number, value: number) {
    if (!pending[name]) pending[name] = [];
    pending[name].push([step, value > 0 ? value : 1e-9]);
    if (!updateScheduled) {
      updateScheduled = true;
      requestAnimationFrame(flush);
    }
  }

  function flush() {
    updateScheduled = false;
    if (!chart) return;
    const names = Object.keys(pending);
    if (names.length === 0) return;
    for (const name of names) {
      ensureSeries(name);
      const items = pending[name];
      delete pending[name];
      const arr = seriesData[name];
      for (let i = 0; i < items.length; i++) arr.push(items[i]);
    }
    const updates = Object.keys(seriesData).map((n) => ({
      name: n,
      data: seriesData[n],
    }));
    chart.setOption(
      { series: updates, legend: { data: Object.keys(seriesData) } },
      { notMerge: false },
    );
  }

  function wsUrlForExperiment(id: string, token: string): string {
    const base =
      env.PUBLIC_API_BASE_URL ||
      (typeof window !== "undefined" ? window.location.origin : "");
    const scheme = base.startsWith("https") ? "wss" : "ws";
    return (
      base.replace(/^http(s)?:/, `${scheme}:`) +
      `/experiments/${id}/logs/stream?token=${encodeURIComponent(token)}&backfill=true`
    );
  }

  async function fetchStreamToken(id: string): Promise<string | null> {
    try {
      const res = await fetch(`/api/experiments/${id}/logs/stream-token`, {
        method: "POST",
      });
      if (!res.ok) return null;
      const data = await res.json();
      return data?.token || null;
    } catch (_) {
      return null;
    }
  }

  function scheduleReconnect() {
    if (reconnectTimer) clearTimeout(reconnectTimer);
    const delay = Math.min(reconnectDelayMs, 10000);
    reconnectTimer = setTimeout(() => {
      reconnectDelayMs = Math.min(reconnectDelayMs * 2, 10000);
      connect();
    }, delay);
  }

  function close() {
    closing = true;
    try {
      ws?.close();
    } catch (_) {}
    ws = null;
    closing = false;
    if (reconnectTimer) clearTimeout(reconnectTimer);
    reconnectTimer = null;
    status = "closed";
  }

  async function connect() {
    if (!experimentId) return;
    if (ws && ws.readyState === WebSocket.OPEN) return;
    status = "connecting";
    const token = await fetchStreamToken(experimentId);
    if (!token) {
      status = "error";
      return;
    }
    const url = wsUrlForExperiment(experimentId, token);
    try {
      ws = new WebSocket(url);
    } catch (_) {
      status = "error";
      scheduleReconnect();
      return;
    }

    ws.onopen = () => {
      reconnectDelayMs = 500;
      status = "open";
    };

    ws.onmessage = (ev: MessageEvent) => {
      try {
        const text =
          typeof ev.data === "string"
            ? ev.data
            : new TextDecoder().decode(ev.data as ArrayBuffer);
        const msg = JSON.parse(text);
        if (msg?.type && msg.type !== "metric") return;
        if (typeof msg?.name === "string" && typeof msg?.value === "number") {
          const name = msg.name as string;
          const val = Number.isFinite(msg.value) ? (msg.value as number) : 0;
          const step =
            typeof msg.step === "number" ? (msg.step as number) : undefined;
          const fallbackIndex = seriesData[name]?.length ?? 0;
          initChart();
          enqueue(name, step ?? fallbackIndex, val);
        }
      } catch (_) {}
    };

    ws.onerror = () => {
      status = "error";
    };

    ws.onclose = () => {
      status = "closed";
      if (!closing) scheduleReconnect();
    };
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
    connect();
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
      close();
      disposeChart();
    };
  });

  // Reconnect if experimentId changes
  $effect(() => {
    const _id = experimentId;
    // reset series/state on experiment change
    seriesData = {};
    pending = {};
    // reapply base axes/dataZoom to avoid index errors
    chart?.setOption(getBaseOptions(), { notMerge: true });
    close();
    connect();
  });
</script>

<div
  class="relative h-80 w-full border border-ctp-surface0/20 bg-transparent overflow-hidden"
>
  <div class="absolute inset-0" bind:this={chartEl}></div>
</div>
