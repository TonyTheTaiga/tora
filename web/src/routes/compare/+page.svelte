<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import Chart from "chart.js/auto";
  import type { PageData } from "./$types";
  import { Circle } from "@lucide/svelte";
  import { reset } from "$lib/state/comparison.svelte.js";
  import { drawBarChart } from "./bar-chart.svelte";
  import { drawScatterChart } from "./scatter-chart.svelte";
  import { drawRadarChart } from "./radar-chart.svelte";
  import { SearchDropdown } from "$lib/components";
  reset();

  let { data }: { data: PageData } = $props();
  let experimentColors_map = $state(new Map<string, string>());
  let selectedMetrics = $state<string[]>([]);
  let searchFilter = $state<string>("");

  let commonMetrics = $derived.by(() => {
    if (!data.experiments?.length) return [];

    const metricSets = data.experiments.map(
      (exp) => new Set(Object.keys(exp.metricData || {})),
    );

    if (metricSets.length === 0) return [];
    let intersection = metricSets[0];
    for (let i = 1; i < metricSets.length; i++) {
      intersection = new Set(
        [...intersection].filter((x) => metricSets[i].has(x)),
      );
    }

    return Array.from(intersection).sort();
  });

  let chartType = $derived(() => {
    if (selectedMetrics.length === 1) return "bar";
    if (selectedMetrics.length === 2) return "scatter";
    if (selectedMetrics.length >= 3) return "radar";
    return "empty";
  });

  let hyperparams = $derived.by(() => {
    const keys = new Set<string>();
    data.experiments?.forEach((exp) =>
      exp.hyperparams?.forEach((hp) => keys.add(hp.key)),
    );
    return Array.from(keys).sort((a, b) => a.localeCompare(b));
  });

  let idToHP = $derived.by(() => {
    const ret = new Map();
    data.experiments?.forEach((exp) => {
      ret.set(exp.id, new Map());
      exp.hyperparams?.forEach((hp) => {
        ret.get(exp.id).set(hp.key, hp.value);
      });
    });
    return ret;
  });

  const catppuccinAccentNames = [
    "--color-ctp-red",
    "--color-ctp-peach",
    "--color-ctp-yellow",
    "--color-ctp-green",
    "--color-ctp-teal",
    "--color-ctp-sky",
    "--color-ctp-sapphire",
    "--color-ctp-blue",
    "--color-ctp-lavender",
    "--color-ctp-mauve",
    "--color-ctp-pink",
    "--color-ctp-flamingo",
    "--color-ctp-rosewater",
    "--color-ctp-maroon",
  ];

  function generateColorVariant(baseColor: string, variant: number): string {
    const match = baseColor.match(/(\d+)/g);
    if (!match || match.length < 3) return baseColor;

    const r = parseInt(match[0]) / 255;
    const g = parseInt(match[1]) / 255;
    const b = parseInt(match[2]) / 255;

    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    let h = 0,
      s = 0,
      l = (max + min) / 2;

    if (max !== min) {
      const d = max - min;
      s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
      switch (max) {
        case r:
          h = (g - b) / d + (g < b ? 6 : 0);
          break;
        case g:
          h = (b - r) / d + 2;
          break;
        case b:
          h = (r - g) / d + 4;
          break;
      }
      h /= 6;
    }

    const saturationAdjust = variant === 1 ? 0.8 : variant === 2 ? 1.2 : 0.9;
    const lightnessAdjust = variant === 1 ? 1.1 : variant === 2 ? 0.9 : 1.05;

    s = Math.min(1, s * saturationAdjust);
    l = Math.min(1, l * lightnessAdjust);

    return `hsl(${Math.round(h * 360)}, ${Math.round(s * 100)}%, ${Math.round(l * 100)}%)`;
  }

  let chartCanvas = $state<HTMLCanvasElement>();
  let chart: Chart | null = null;

  function updateChart() {
    if (chart) {
      chart.destroy();
      chart = null;
    }

    if (
      !chartCanvas ||
      chartType() === "empty" ||
      !data.experiments ||
      experimentColors_map.size === 0
    ) {
      return;
    }

    switch (chartType()) {
      case "bar":
        chart = drawBarChart(
          chartCanvas,
          data.experiments,
          selectedMetrics[0],
          experimentColors_map,
        );
        break;
      case "scatter":
        chart = drawScatterChart(
          chartCanvas,
          data.experiments,
          selectedMetrics as [string, string],
          experimentColors_map,
        );
        break;
      case "radar":
        chart = drawRadarChart(
          chartCanvas,
          data.experiments,
          selectedMetrics,
          experimentColors_map,
        );
        break;
    }
  }

  $effect(() => {
    updateChart();
  });

  onDestroy(() => {
    if (chart) {
      chart.destroy();
    }
  });

  onMount(() => {
    const resolveAndSetColors = () => {
      if (!data.experiments) return;

      const computedStyles = getComputedStyle(document.documentElement);
      const newColorMap = new Map<string, string>();
      const resolvedBaseColors = catppuccinAccentNames.map((name) =>
        computedStyles.getPropertyValue(name).trim(),
      );

      data.experiments.forEach((exp, index) => {
        const numBaseColors = resolvedBaseColors.length;

        if (index < numBaseColors) {
          newColorMap.set(exp.id, resolvedBaseColors[index]);
        } else {
          const baseIndex = (index - numBaseColors) % numBaseColors;
          const variant =
            Math.floor((index - numBaseColors) / numBaseColors) + 1;
          const resolvedBaseColor = resolvedBaseColors[baseIndex];
          const variantColor = generateColorVariant(resolvedBaseColor, variant);
          newColorMap.set(exp.id, variantColor);
        }
      });

      experimentColors_map = newColorMap;
    };

    resolveAndSetColors();
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    mediaQuery.addEventListener("change", resolveAndSetColors);
    return () => {
      mediaQuery.removeEventListener("change", resolveAndSetColors);
    };
  });
</script>

<div class="font-mono">
  <!-- Header -->
  <div
    class="flex items-center justify-between p-4 md:p-6 border-b border-ctp-surface0/10"
  >
    <div
      class="flex items-stretch gap-3 md:gap-4 min-w-0 flex-1 pr-4 min-h-fit"
    >
      <div
        class="w-2 bg-ctp-blue rounded-full flex-shrink-0 self-stretch"
      ></div>
      <div class="min-w-0 flex-1 py-1">
        <h1 class="text-lg md:text-xl text-ctp-text truncate font-mono">
          Experiment Comparison
        </h1>
        <div class="text-sm text-ctp-subtext0 space-y-1">
          <div>
            {data.experiments?.length || 0} experiments selected
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="px-4 md:px-6 py-6 space-y-6">
    <!-- Legend -->
    <div>
      <div class="text-sm text-ctp-text mb-3">Legend</div>
      <div class="space-y-1">
        <!-- Header -->
        <div
          class="flex items-center text-sm text-ctp-subtext0 pb-1 border-b border-ctp-surface0/20"
        >
          <div class="flex-1">name</div>
          <div class="w-20 text-right">color</div>
        </div>
        <!-- Experiment entries -->
        {#each data.experiments as experiment}
          <div
            class="flex items-center text-sm hover:bg-ctp-surface0/10 px-1 py-1 transition-colors"
          >
            <div
              class="flex-1 text-ctp-text truncate min-w-0"
              title={experiment.name}
            >
              {experiment.name}
            </div>
            <div class="w-20 text-right">
              <Circle
                size={8}
                style="color: {experimentColors_map.get(
                  experiment.id,
                )}; fill: {experimentColors_map.get(experiment.id)};"
                class="inline"
              />
            </div>
          </div>
        {/each}
      </div>
    </div>

    <!-- Hyperparameters -->
    <div>
      <div class="text-sm text-ctp-text mb-3">Hyperparameters</div>
      <div
        class="bg-ctp-surface0/10 border border-ctp-surface0/20 overflow-hidden"
      >
        <div
          class="overflow-x-auto overflow-y-auto max-h-60 scroll-container"
          style="scrollbar-width: none; -ms-overflow-style: none;"
        >
          <table class="w-full text-sm text-left font-mono">
            <thead class="sticky top-0 z-10 bg-ctp-mantle">
              <tr>
                <th
                  class="border-b border-ctp-surface0/20 sticky left-0 z-20 bg-ctp-mantle px-2 py-1 w-4"
                ></th>
                {#each hyperparams as hyperparam}
                  <th
                    class="px-2 py-1 text-ctp-subtext0 border-b border-ctp-surface0/20 whitespace-nowrap"
                  >
                    {hyperparam}
                  </th>
                {/each}
              </tr>
            </thead>
            <tbody>
              {#each data.experiments as experiment}
                <tr class="hover:bg-ctp-surface0/20 transition-colors">
                  <td
                    class="bg-ctp-mantle sticky left-0 z-10 px-2 py-1 text-center border-r border-ctp-surface0/20"
                  >
                    <Circle
                      size={8}
                      style="color: {experimentColors_map.get(
                        experiment.id,
                      )}; fill: {experimentColors_map.get(experiment.id)};"
                    />
                  </td>
                  {#each hyperparams as key}
                    <td class="px-2 py-1 text-ctp-text whitespace-nowrap">
                      {idToHP.get(experiment.id).get(key) ?? "-"}
                    </td>
                  {/each}
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- Metrics Comparison -->
    <div>
      <div class="text-sm text-ctp-text mb-3">Metrics Comparison</div>
      <div class="bg-ctp-surface0/10 border border-ctp-surface0/20">
        <!-- Metric Selector -->
        <div class="p-3 border-b border-ctp-surface0/20">
          <SearchDropdown
            items={commonMetrics}
            bind:selectedItems={selectedMetrics}
            bind:searchQuery={searchFilter}
            getItemText={(metric) => metric}
            itemTypeName="metrics"
            placeholder="filter metrics..."
          />
        </div>

        <div class="p-4">
          {#if chartType() !== "empty"}
            <div
              class={chartType() === "radar"
                ? "aspect-square max-w-2xl mx-auto"
                : "aspect-[4/3] max-w-4xl mx-auto"}
            >
              <canvas bind:this={chartCanvas} class="w-full h-full chart-canvas"
              ></canvas>
            </div>
          {:else}
            <div
              class="aspect-square max-w-2xl mx-auto flex items-center justify-center"
            >
              <div class="text-center text-ctp-subtext0">
                <div class="text-sm mb-2">select metrics</div>
                <div class="text-sm text-ctp-subtext1">no data to display</div>
              </div>
            </div>
          {/if}
        </div>
      </div>
    </div>
  </div>
</div>

<style>
  .scroll-container::-webkit-scrollbar {
    display: none;
  }

  .chart-canvas {
    touch-action: none;
    user-select: none;
    -webkit-user-select: none;
    -webkit-touch-callout: none;
  }
</style>
