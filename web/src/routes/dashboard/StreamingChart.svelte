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
  import type {} from "svelte";

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
  let status = $state<"idle" | "connecting" | "open" | "closed" | "error">(
    "idle",
  );
  let yScale = $state<"log" | "linear">("log");
  export function toggleScale() {
    yScale = yScale === "log" ? "linear" : "log";
    // apply immediately on toggle to avoid extra effects
    applyTheme();
  }

  let chartEl: HTMLDivElement | null = $state(null);
  let chart: EChartsType | null = null;
  let seriesData: Record<string, Array<[number, number]>> = {};
  let pending: Record<string, Array<[number, number]>> = {};
  let updateScheduled = false;
  let seenMsgIds: Set<string> = new Set();

  let ws: WebSocket | null = null;
  let reconnectDelayMs = 500;
  let closing = false;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

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
        id: n,
        name: n,
        type: "line",
        showSymbol: false,
        smooth: 0.15,
        connectNulls: true,
        data:
          yScale === "log"
            ? seriesData[n].map(([x, y]) => [x, y > 0 ? y : null])
            : seriesData[n].map(([x, y]) => [x, Number.isFinite(y) ? y : null]),
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
    pending[name].push([step, value]);
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
      // Sort by step to ensure continuous lines; no step-based dedupe
      seriesData[name] = arr.slice().sort((a, b) => a[0] - b[0]);
    }
    const updates = Object.keys(seriesData).map((n) => ({
      id: n,
      name: n,
      type: "line",
      showSymbol: false,
      smooth: 0.15,
      connectNulls: true,
      emphasis: { focus: "series" },
      data:
        yScale === "log"
          ? seriesData[n].map(([x, y]) => [x, y > 0 ? y : null])
          : seriesData[n].map(([x, y]) => [x, Number.isFinite(y) ? y : null]),
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

  export function refreshChart() {
    // Reset series/state and reconnect
    seriesData = {};
    pending = {};
    seenMsgIds = new Set();
    chart?.setOption(getBaseOptions(), { notMerge: true });
    close();
    connect();
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
        // Deduplicate by message id, if provided
        const _md: any = (msg as any)?.metadata;
        const msgId: unknown =
          (msg as any)?.msg_id ?? _md?.msg_id ?? _md?.message_id;
        if (typeof msgId === "string") {
          if (seenMsgIds.has(msgId)) return;
          seenMsgIds.add(msgId);
        }
        if (typeof msg?.name === "string" && typeof msg?.value === "number") {
          const name = msg.name as string;
          const val = Number.isFinite(msg.value) ? (msg.value as number) : 0;
          const step =
            typeof msg.step === "number" ? (msg.step as number) : undefined;
          const fallbackIndex = seriesData[name]?.length ?? 0;
          if (!chart) initChart();
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
    const updates = Object.keys(seriesData).map((n) => ({
      id: n,
      data:
        yScale === "log"
          ? seriesData[n].map(([x, y]) => [x, y > 0 ? y : null])
          : seriesData[n].map(([x, y]) => [x, Number.isFinite(y) ? y : null]),
    }));
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
      close();
      disposeChart();
    };
  });

  // Reconnect/reset when experiment changes
  $effect(() => {
    const id = experimentId;
    if (!id) return;
    // reset series/state on experiment change
    seriesData = {};
    pending = {};
    seenMsgIds = new Set();
    // reapply base axes/dataZoom to avoid index errors
    chart?.setOption(getBaseOptions(), { notMerge: true });
    close();
    connect();
  });
</script>

<div
  class="relative h-80 w-full bg-transparent border border-ctp-surface0/20 overflow-hidden"
>
  <div class="absolute inset-0" bind:this={chartEl}></div>
</div>
