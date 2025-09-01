<script lang="ts">
  import ExperimentChart from "./ExperimentChart.svelte";
  import type { Experiment, HyperParam } from "$lib/types";
  import { copyToClipboard } from "$lib/utils/common";
  import { loading, errors } from "./state.svelte";
  import { ChevronDown, ChevronRight, Pin } from "@lucide/svelte";
  import { env } from "$env/dynamic/public";

  let { experiment } = $props();
  let results = $state<any[]>([]);
  let metricData = $state<Record<string, number[]>>({});
  let pinnedNames = $state<string[]>([]);
  let showResults = $state(true);
  let showHeader = $state(true);
  let showAllHyperparams = $state(false);

  let sortedHyperparams = $derived(
    experiment.hyperparams
      ?.slice()
      .sort((a: HyperParam, b: HyperParam) => a.key.localeCompare(b.key)) ?? [],
  );

  let pinnedResults = $derived(
    pinnedNames
      .map((name) => results.find((r) => r.name === name))
      .filter((m): m is any => Boolean(m)),
  );

  function storageKey(expId: string) {
    return `tora:pinnedResults:${expId}`;
  }

  function headerExpandedKey(expId: string) {
    return `tora:headerExpanded:${expId}`;
  }

  function loadPinned() {
    try {
      if (typeof localStorage === "undefined") {
        pinnedNames = [];
        return;
      }
      const raw = localStorage.getItem(storageKey(experiment.id));
      if (!raw) {
        pinnedNames = [];
        return;
      }
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) {
        pinnedNames = parsed.filter((x) => typeof x === "string");
      } else {
        pinnedNames = [];
      }
    } catch (e) {
      pinnedNames = [];
    }
  }

  function savePinned() {
    try {
      if (typeof localStorage === "undefined") return;
      localStorage.setItem(
        storageKey(experiment.id),
        JSON.stringify(pinnedNames),
      );
    } catch (e) {}
  }

  function isPinned(name: string): boolean {
    return pinnedNames.includes(name);
  }

  function togglePin(name: string) {
    if (isPinned(name)) {
      pinnedNames = pinnedNames.filter((n) => n !== name);
    } else {
      pinnedNames = [...pinnedNames, name];
    }
    savePinned();
  }

  function displayHPValue(v: string | number): string {
    const s = String(v);
    return s.length > 16 ? s.slice(0, 16) + "…" : s;
  }

  function loadHeaderExpanded() {
    try {
      if (typeof localStorage === "undefined") return;
      const raw = localStorage.getItem(headerExpandedKey(experiment.id));
      showHeader = raw === null ? true : raw === "true";
    } catch (e) {}
  }

  function saveHeaderExpanded() {
    try {
      if (typeof localStorage === "undefined") return;
      localStorage.setItem(
        headerExpandedKey(experiment.id),
        String(showHeader),
      );
    } catch (e) {}
  }

  // --- Live stream (WebSocket) ---
  let ws: WebSocket | null = null;
  let reconnectDelayMs = 500;
  let closing = false;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let liveStreaming = $state(false);
  let wsStatus = $state<"idle" | "connecting" | "open" | "closed" | "error">(
    "idle",
  );

  function wsUrlForExperiment(id: string, token: string): string {
    const base =
      env.PUBLIC_API_BASE_URL ||
      (typeof window !== "undefined" ? window.location.origin : "");
    const scheme = base.startsWith("https") ? "wss" : "ws";
    return (
      base.replace(/^http(s)?:/, `${scheme}:`) +
      `/api/experiments/${id}/logs/stream?token=${encodeURIComponent(token)}`
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
    } catch (e) {
      return null;
    }
  }

  function scheduleReconnect() {
    if (reconnectTimer) clearTimeout(reconnectTimer);
    if (!liveStreaming) return; // only reconnect if user wants streaming
    const delay = Math.min(reconnectDelayMs, 10000);
    reconnectTimer = setTimeout(() => {
      if (!liveStreaming) return;
      reconnectDelayMs = Math.min(reconnectDelayMs * 2, 10000);
      connectStream();
    }, delay);
  }

  function closeStream() {
    closing = true;
    try {
      ws?.close();
    } catch (_) {}
    ws = null;
    closing = false;
    if (reconnectTimer) clearTimeout(reconnectTimer);
    reconnectTimer = null;
    wsStatus = "closed";
  }

  async function connectStream() {
    if (typeof window === "undefined" || !experiment?.id) return;
    if (ws && ws.readyState === WebSocket.OPEN) return;
    wsStatus = "connecting";
    const token = await fetchStreamToken(experiment.id);
    if (!token) {
      // No token, no live updates (user may not be authorized)
      wsStatus = "error";
      return;
    }

    const url = wsUrlForExperiment(experiment.id, token);
    try {
      ws = new WebSocket(url);
    } catch (e) {
      wsStatus = "error";
      scheduleReconnect();
      return;
    }

    ws.onopen = () => {
      reconnectDelayMs = 500;
      wsStatus = "open";
    };

    ws.onmessage = (ev: MessageEvent) => {
      try {
        const text =
          typeof ev.data === "string"
            ? ev.data
            : new TextDecoder().decode(ev.data as ArrayBuffer);
        const msg = JSON.parse(text);
        if (
          msg?.type === "metric" &&
          typeof msg.name === "string" &&
          typeof msg.value === "number"
        ) {
          const name = msg.name;
          const val = Number.isFinite(msg.value) ? msg.value : 0;
          const step = typeof msg.step === "number" ? msg.step : undefined;
          // Prefer imperative append to avoid rerendering
          try {
            chartRef?.appendPoint?.(name, step ?? 0, val);
          } catch (_) {
            // Fallback: minimal state bump (may rerender)
            const next = { ...metricData } as Record<string, number[]>;
            const series = next[name] ? next[name].slice() : [];
            series.push(val);
            next[name] = series;
            metricData = next;
          }
        }
      } catch (_) {
        // ignore parse errors
      }
    };

    ws.onerror = () => {
      wsStatus = "error";
      // let onclose decide about reconnect
    };

    ws.onclose = () => {
      wsStatus = "closed";
      if (!closing && liveStreaming) scheduleReconnect();
    };
  }

  async function toggleLiveStream() {
    if (liveStreaming) {
      liveStreaming = false;
      closeStream();
    } else {
      liveStreaming = true;
      // Reload historical data to avoid gaps before starting the live stream
      try {
        await loadExperimentDetails(experiment);
      } catch (_) {}
      await connectStream();
    }
  }

  async function loadExperimentDetails(experiment: Experiment) {
    try {
      loading.experimentDetails = true;
      errors.experimentDetails = null;
      const response = await fetch(`/api/experiments/${experiment.id}/logs`);
      if (!response.ok)
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      const apiResponse = await response.json();
      const metrics = apiResponse.data;
      if (!metrics || !Array.isArray(metrics))
        throw new Error("Invalid response structure from metrics API");

      const resultsByName = new Map<string, any>();
      const seriesByName = new Map<string, any[]>();

      metrics.forEach((m: any) => {
        const mType = m?.metadata?.type ?? "metric";
        if (mType === "result") {
          resultsByName.set(m.name, m);
        } else {
          if (!seriesByName.has(m.name)) seriesByName.set(m.name, []);
          seriesByName.get(m.name)!.push(m);
        }
      });

      const scalarMetricsList: any[] = Array.from(resultsByName.values());
      const computedMetricData: Record<string, number[]> = {};

      seriesByName.forEach((metricList, name) => {
        computedMetricData[name] = metricList
          .sort((a, b) => (a.step || 0) - (b.step || 0))
          .map((m) => m.value);
      });

      results = scalarMetricsList;
      metricData = computedMetricData;
    } catch (error) {
      errors.experimentDetails =
        error instanceof Error
          ? error.message
          : "Failed to load experiment details";
    } finally {
      loading.experimentDetails = false;
    }
  }

  $effect(() => {
    loadExperimentDetails(experiment);
  });

  // Reconnect WS when experiment changes if streaming is enabled; cleanup on unmount
  $effect(() => {
    if (!experiment?.id) return;
    if (liveStreaming) {
      closeStream();
      connectStream();
    }
    return () => closeStream();
  });

  $effect(() => {
    loadPinned();
    loadHeaderExpanded();
  });

  $effect(() => {
    if (!results || results.length === 0) return;
    const available = new Set(results.map((r) => r.name));
    const pruned = pinnedNames.filter((n) => available.has(n));
    if (pruned.length !== pinnedNames.length) {
      pinnedNames = pruned;
      savePinned();
    }
  });

  $effect(() => {
    const _h = showHeader;
    saveHeaderExpanded();
  });
</script>

<div class="flex flex-col">
  <div
    class="sticky top-0 z-10 surface-elevated border-b border-ctp-surface0/30 p-4"
  >
    <div class="mb-1">
      <button
        class="flex items-center gap-2 text-ctp-text font-medium text-base hover:text-ctp-blue"
        onclick={() => (showHeader = !showHeader)}
        aria-expanded={showHeader}
        aria-controls="experiment-header-details"
      >
        {#if showHeader}
          <ChevronDown size={16} />
        {:else}
          <ChevronRight size={16} />
        {/if}
        <span class="truncate">{experiment.name}</span>
      </button>
    </div>
    {#if showHeader}
      <div id="experiment-header-details" class="mt-2">
        {#if experiment.description}
          <p class="text-ctp-subtext0 mb-2 text-sm">
            {experiment.description}
          </p>
        {/if}
        {#if experiment.tags?.length}
          <div class="flex flex-wrap gap-1 mb-2">
            {#each experiment.tags as tag}
              <span
                class="text-xs bg-ctp-blue/20 text-ctp-blue border border-ctp-blue/30 px-2 py-1"
                >{tag}</span
              >
            {/each}
          </div>
        {/if}

        {#if experiment.hyperparams?.length}
          <div
            id="hyperparams-header-section"
            class="border-ctp-terminal-border"
          >
            <div class="flex flex-wrap gap-1">
              {#each showAllHyperparams ? sortedHyperparams : sortedHyperparams.slice(0, 12) as param}
                <span
                  class="text-[11px] rounded-sm bg-ctp-surface0/30 border border-ctp-surface0/40 text-ctp-subtext0 font-mono px-2 py-0.5"
                  title={`${param.key}=${String(param.value)}`}
                >
                  {param.key}={displayHPValue(param.value)}
                </span>
              {/each}
              {#if !showAllHyperparams && sortedHyperparams.length > 12}
                <button
                  class="text-[11px] text-ctp-blue px-2 py-0.5"
                  onclick={() => (showAllHyperparams = true)}
                >
                  +{sortedHyperparams.length - 12} more
                </button>
              {:else if showAllHyperparams && sortedHyperparams.length > 12}
                <button
                  class="text-[11px] text-ctp-blue px-2 py-0.5"
                  onclick={() => (showAllHyperparams = false)}
                >
                  show less
                </button>
              {/if}
            </div>
          </div>
        {/if}
      </div>
    {/if}

    <div class="flex items-center gap-3 mb-2">
      <button
        class="text-xs text-ctp-overlay0 hover:text-ctp-blue"
        onclick={() => copyToClipboard(experiment.id)}
        title="click to copy experiment id"
      >
        experiment id: {experiment.id}
      </button>

      <button
        class="text-xs border border-ctp-surface0/40 px-2 py-1 hover:text-ctp-blue"
        onclick={toggleLiveStream}
        aria-pressed={liveStreaming}
        title={liveStreaming ? "stop live stream" : "start live stream"}
      >
        {liveStreaming ? "stop live stream" : "start live stream"}
      </button>

      <span class="text-[11px] text-ctp-subtext0">
        live: {wsStatus}
      </span>
    </div>
  </div>

  <div class="p-4">
    {#if loading.experimentDetails}
      <div class="text-center py-12">
        <div class="text-ctp-subtext0 text-sm">
          loading experiment details...
        </div>
      </div>
    {:else if errors.experimentDetails}
      <div class="surface-layer-2 p-4">
        <div class="text-ctp-red font-medium text-sm mb-3">
          error loading experiment details
        </div>
        <div class="text-ctp-subtext0 mb-4 text-xs">
          {errors.experimentDetails}
        </div>
        >
      </div>
    {:else}
      <div class="space-y-6">
        {#if pinnedResults.length > 0}
          <div class="space-y-2">
            <div class="flex items-center gap-2">
              <div class="text-sm text-ctp-text">pinned results</div>
              <div class="text-sm text-ctp-subtext0">
                [{pinnedResults.length}]
              </div>
            </div>
            <div
              class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3"
            >
              {#each pinnedResults as metric}
                <div
                  class="relative flex flex-col gap-1 p-3 border-ctp-terminal-border hover:bg-ctp-surface0/20 min-w-0"
                >
                  <button
                    class="absolute top-2 right-2 text-ctp-overlay0 hover:text-ctp-yellow"
                    title="unpin result"
                    aria-label="unpin result"
                    onclick={() => togglePin(metric.name)}
                  >
                    <Pin size={14} />
                  </button>
                  <div
                    class="text-ctp-subtext0 text-[11px] uppercase tracking-wide truncate pr-6"
                    title={metric.name}
                  >
                    {metric.name}
                  </div>
                  <div
                    class="text-ctp-text font-semibold tabular-nums font-mono text-lg truncate"
                    title={String(metric.value)}
                  >
                    {typeof metric.value === "number"
                      ? metric.value.toFixed(4)
                      : metric.value}
                  </div>
                </div>
              {/each}
            </div>
          </div>
        {/if}

        {#if Object.keys(metricData).length > 0}
          <div class="space-y-2">
            <ExperimentChart bind:this={chartRef} {metricData} />
          </div>
        {/if}

        {#if results.length > 0}
          <div class="space-y-2">
            <button
              class="flex items-center gap-2 text-sm text-ctp-text hover:text-ctp-blue"
              onclick={() => (showResults = !showResults)}
              aria-expanded={showResults}
              aria-controls="results-section"
            >
              {#if showResults}
                <ChevronDown size={14} />
              {:else}
                <ChevronRight size={14} />
              {/if}
              <span>results</span>
              <span class="text-ctp-subtext0">[{results.length}]</span>
            </button>
            {#if showResults}
              <div id="results-section" class="border-ctp-terminal-border p-2">
                <div
                  class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3"
                >
                  {#each results as metric}
                    <div
                      class="relative flex flex-col gap-1 p-2 hover:bg-ctp-surface0/20 min-w-0"
                    >
                      <button
                        class="absolute top-2 right-2 text-ctp-overlay0 hover:text-ctp-yellow"
                        title={isPinned(metric.name) ? "unpin" : "pin"}
                        aria-pressed={isPinned(metric.name)}
                        onclick={() => togglePin(metric.name)}
                      >
                        <Pin
                          size={14}
                          class={isPinned(metric.name) ? "text-ctp-yellow" : ""}
                        />
                      </button>
                      <div
                        class="text-ctp-subtext0 text-[11px] uppercase tracking-wide truncate pr-6"
                        title={metric.name}
                      >
                        {metric.name}
                      </div>
                      <div
                        class="text-ctp-text font-semibold tabular-nums font-mono truncate pr-6"
                        title={String(metric.value)}
                      >
                        {typeof metric.value === "number"
                          ? metric.value.toFixed(4)
                          : metric.value}
                      </div>
                    </div>
                  {/each}
                </div>
              </div>
            {/if}
          </div>
        {/if}
      </div>
    {/if}
  </div>
</div>
