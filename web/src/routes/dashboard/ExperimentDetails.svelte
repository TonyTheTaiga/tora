<script lang="ts">
  import StaticChart from "./StaticChart.svelte";
  import StreamingChart from "./StreamingChart.svelte";
  import type { Experiment, HyperParam } from "$lib/types";
  import { copyToClipboard } from "$lib/utils/common";
  import { loading, errors } from "./state.svelte";
  import { ChevronDown, ChevronRight, Pin, RefreshCw } from "@lucide/svelte";
  import {
    loadPins,
    togglePin as togglePinGlobal,
    isPinned as isPinnedGlobal,
  } from "./pins.svelte";
  import { onMount } from "svelte";

  let { experiment } = $props();
  let results = $state<any[]>([]);
  let isStreamingChart = $state(false);
  let showResults = $state(true);
  let showHeader = $state(true);
  let showAllHyperparams = $state(false);
  let pinnedResults = $state<any[]>([]);

  let sortedHyperparams = $derived(
    experiment.hyperparams
      ?.slice()
      .sort((a: HyperParam, b: HyperParam) => a.key.localeCompare(b.key)) ?? [],
  );

  function headerExpandedKey(expId: string) {
    return `tora:headerExpanded:${expId}`;
  }

  function storageKey(expId: string) {
    return `tora:pinnedResults:${expId}`;
  }
  function loadPinnedResults() {
    try {
      if (typeof localStorage === "undefined") {
        pinnedResults = [];
        return;
      }
      const raw = localStorage.getItem(storageKey(experiment.id));
      if (!raw) {
        pinnedResults = [];
        return;
      }
      const names = JSON.parse(raw);
      if (!Array.isArray(names)) {
        pinnedResults = [];
        return;
      }
      const allow = new Set(
        names.filter((n: unknown) => typeof n === "string") as string[],
      );
      pinnedResults = results.filter((r: any) => allow.has(r?.name));
    } catch (_) {
      pinnedResults = [];
    }
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

  function toggleLiveStream() {
    if (isStreamingChart) {
      isStreamingChart = false;
    } else {
      isStreamingChart = true;
    }
  }

  async function loadExperimentDetails(experiment: Experiment) {
    try {
      loading.experimentDetails = true;
      errors.experimentDetails = null;
      const response = await fetch(
        `/api/experiments/${experiment.id}/results`,
        {
          signal: detailsAbort?.signal,
        },
      );
      if (!response.ok)
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      const apiResponse = await response.json();
      const metrics = apiResponse.data;
      if (!metrics || !Array.isArray(metrics))
        throw new Error("Invalid response structure from metrics API");

      const seen = new Set<string>();
      const list: any[] = [];
      for (const m of metrics) {
        if (!seen.has(m.name)) {
          list.push(m);
          seen.add(m.name);
        }
      }
      results = list;
      loadPinnedResults();
    } catch (error) {
      errors.experimentDetails =
        error instanceof Error
          ? error.message
          : "Failed to load experiment details";
    } finally {
      loading.experimentDetails = false;
    }
  }

  let detailsAbort: AbortController | null = null;

  onMount(() => {
    const exp = experiment;
    if (exp) {
      try {
        detailsAbort?.abort();
      } catch {}
      detailsAbort = new AbortController();
      loadPins(exp.id);
      loadExperimentDetails(exp);
      loadHeaderExpanded();
    }
    return () => {
      try {
        detailsAbort?.abort();
      } catch {}
      detailsAbort = null;
    };
  });

  async function refreshDetails() {
    try {
      errors.experimentDetails = null;
      try {
        detailsAbort?.abort();
      } catch {}
      detailsAbort = new AbortController();
      await loadExperimentDetails(experiment);
    } catch (_) {
      // errors handled inside loadExperimentDetails
    }
  }

  function toggleHeader() {
    showHeader = !showHeader;
    saveHeaderExpanded();
  }

  function isPinned(name: string): boolean {
    return isPinnedGlobal(experiment.id, name);
  }

  function togglePin(name: string) {
    togglePinGlobal(experiment.id, name);
    loadPinnedResults();
  }
</script>

<div class="flex flex-col">
  <div
    class="sticky top-0 z-10 surface-elevated border-b border-ctp-surface0/30 p-4"
  >
    <div class="flex items-center justify-between mb-2">
      <button
        class="flex items-center gap-2 text-ctp-text font-medium text-base hover:text-ctp-blue"
        onclick={toggleHeader}
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
      <button
        class="inline-flex items-center gap-1 text-xs bg-transparent border border-ctp-surface0/40 hover:border-ctp-surface0/60 text-ctp-overlay0 hover:text-ctp-text px-2 py-1 transition-colors disabled:opacity-50"
        onclick={refreshDetails}
        disabled={loading.experimentDetails}
        title="refresh experiment details"
      >
        <RefreshCw class="w-3.5 h-3.5" />
        refresh
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
    </div>
  </div>

  <div class="p-4">
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
      <div class="space-y-2">
        {#if isStreamingChart}
          <StreamingChart experimentId={experiment.id}>
            {#snippet toggleStreaming()}
              <button
                class="text-[10px] leading-none border border-ctp-surface0/40 px-1.5 py-0.5 bg-ctp-mantle/70 hover:text-ctp-yellow"
                onclick={toggleLiveStream}
                aria-pressed={true}
                title="stop live stream"
              >
                stop live
              </button>
            {/snippet}
          </StreamingChart>
        {:else}
          <StaticChart experimentId={experiment.id}>
            {#snippet toggleStreaming()}
              <button
                class="text-[10px] leading-none border border-ctp-surface0/40 px-1.5 py-0.5 bg-ctp-mantle/70 hover:text-ctp-blue"
                onclick={toggleLiveStream}
                aria-pressed={false}
                title="start live stream"
              >
                start live
              </button>
            {/snippet}
          </StaticChart>
        {/if}
      </div>

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
            {#if loading.experimentDetails}
              <div class="text-center py-8 text-ctp-subtext0 text-sm">
                loading experiment details...
              </div>
            {:else if errors.experimentDetails}
              <div class="surface-layer-2 p-4">
                <div class="text-ctp-red font-medium text-sm mb-3">
                  error loading experiment details
                </div>
                <div class="text-ctp-subtext0 mb-4 text-xs">
                  {errors.experimentDetails}
                </div>
              </div>
            {:else if results.length > 0}
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
            {/if}
          </div>
        {/if}
      </div>
    </div>
  </div>
</div>
