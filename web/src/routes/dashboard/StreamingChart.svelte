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
  import { env } from "$env/dynamic/public";
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

  let { experimentId, yScale } = $props<{
    experimentId: string;
    yScale: "log" | "linear";
  }>();
  let status = $state<"idle" | "connecting" | "open" | "closed" | "error">(
    "idle",
  );
  let chartEl: HTMLDivElement | null = null;
  let chart: EChartsType | null = null;
  let ro: ResizeObserver | null = null;
  let seriesData: Record<string, Array<[number, number]>> = {};
  let pending: Record<string, Array<[number, number]>> = {};
  let updateScheduled = false;
  let seenMsgIds: Set<string> = new Set();

  let ws: WebSocket | null = null;
  let reconnectDelayMs = 500;
  let closing = false;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let ac: AbortController | null = null;
  let chartTheme = $state(getChartTheme());

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

  function ensureSeries(name: string) {
    if (!seriesData[name]) {
      seriesData[name] = [];
      const names = Object.keys(seriesData);
      const newSeries = lineSeriesFrom(transformForScale(seriesData, yScale));
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
      seriesData[name] = arr.slice().sort((a, b) => a[0] - b[0]);
    }
    const updates = lineSeriesFrom(transformForScale(seriesData, yScale));
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

  async function fetchStreamToken(
    id: string,
    signal?: AbortSignal,
  ): Promise<string | null> {
    try {
      const res = await fetch(`/api/experiments/${id}/logs/stream-token`, {
        method: "POST",
        signal,
      });
      if (!res.ok) return null;
      const data = await res.json();
      return data?.token || null;
    } catch (_) {
      return null;
    }
  }

  function scheduleReconnect() {
    if (ac?.signal.aborted) return;
    if (reconnectTimer) clearTimeout(reconnectTimer);
    const delay = Math.min(reconnectDelayMs, 10000);
    reconnectTimer = setTimeout(() => {
      if (ac?.signal.aborted) return;
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
    if (ac?.signal.aborted) return;
    if (ws && ws.readyState === WebSocket.OPEN) return;
    status = "connecting";
    const token = await fetchStreamToken(experimentId, ac?.signal);
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
      data: transformForScale(seriesData, yScale)[n],
    }));
    chart?.setOption(
      { ...themeAxisUpdate(chartTheme, yScale), series: updates },
      { notMerge: false },
    );
  }

  onMount(() => {
    ac = new AbortController();
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
    connect();

    return async () => {
      try {
        ac?.abort();
      } catch {}
      ac = null;
      if (mediaQuery)
        mediaQuery.removeEventListener("change", handleThemeChange);
      if (observer) observer.disconnect();
      if (ro) {
        try {
          ro.disconnect();
        } catch {}
        ro = null;
      }
      const waitForClose = () =>
        new Promise<void>((resolve) => {
          if (!ws || ws.readyState === WebSocket.CLOSED) return resolve();
          try {
            const handler = () => resolve();
            ws.addEventListener("close", handler, { once: true });
            ws.close();
          } catch (_) {
            resolve();
          }
        });
      try {
        await waitForClose();
      } finally {
        close();
        await Promise.resolve();
        disposeChart();
      }
    };
  });
</script>

<div
  class="relative h-80 w-full bg-transparent border border-ctp-surface0/20 overflow-hidden"
>
  <div class="absolute inset-0 py-2" bind:this={chartEl}></div>
</div>
