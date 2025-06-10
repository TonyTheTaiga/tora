<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import Chart from "chart.js/auto";
  import type { PageData } from "./$types";
  import { Circle, ChevronDown } from "lucide-svelte";
  import { reset } from "$lib/state/comparison.svelte.js";
  import { drawBarChart } from "./bar-chart.svelte";
  import { drawScatterChart } from "./scatter-chart.svelte";
  import { drawRadarChart } from "./radar-chart.svelte";
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

  let filteredMetrics = $derived(
    commonMetrics.filter((metric) =>
      metric.toLowerCase().includes(searchFilter.toLowerCase()),
    ),
  );

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

  function selectAllMetrics() {
    selectedMetrics = [...commonMetrics];
  }

  function clearAllMetrics() {
    selectedMetrics = [];
  }

  function toggleMetric(metric: string) {
    if (selectedMetrics.includes(metric)) {
      selectedMetrics = selectedMetrics.filter((m) => m !== metric);
    } else {
      selectedMetrics = [...selectedMetrics, metric];
    }
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

<div class="text-ctp-text">
  <div class="mb-4 mt-8">
    <!-- Legend -->
    <div class="mb-4 p-3 pt-4">
      <h4 class="text-xs font-medium text-ctp-subtext1 mb-3">Legend</h4>
      <div class="flex flex-wrap gap-x-4 gap-y-2 overflow-x-auto">
        {#each data.experiments as experiment}
          <div class="flex items-center gap-2 min-w-0 flex-shrink-0">
            <Circle
              size={10}
              style="color: {experimentColors_map.get(
                experiment.id,
              )}; fill: {experimentColors_map.get(experiment.id)};"
              class="flex-shrink-0"
            />
            <span
              class="text-xs text-ctp-text truncate max-w-40"
              title={experiment.name}>{experiment.name}</span
            >
          </div>
        {/each}
      </div>
    </div>
  </div>

  <div class="mb-4 mt-8">
    <!-- hyperparams -->
    <div class="text-sm font-semibold text-ctp-text p-4 border-ctp-surface1">
      Hyperparameters
    </div>
    <div
      class="overflow-x-auto overflow-y-auto max-h-44 scroll-container"
      style="scrollbar-width: none; -ms-overflow-style: none;"
    >
      <table class="w-full text-sm text-left">
        <thead class="sticky top-0 z-10">
          <tr>
            <th
              class="bg-ctp-crust border-b border-ctp-surface0 sticky left-0 z-20"
            >
            </th>
            {#each hyperparams as hyperparam}
              <th
                class="p-3 font-medium text-ctp-subtext1 border-b border-ctp-surface0 whitespace-nowrap"
              >
                {hyperparam.toLowerCase()}
              </th>
            {/each}
          </tr>
        </thead>
        <tbody>
          {#each data.experiments as experiment}
            <tr
              class="border-t border-ctp-surface0 hover:bg-ctp-surface0/50 transition-colors duration-200"
            >
              <th
                scope="row"
                class="bg-ctp-crust p-3 text-ctp-text font-medium sticky left-0 text-center"
              >
                <Circle
                  size={16}
                  style="color: {experimentColors_map.get(
                    experiment.id,
                  )}; fill: {experimentColors_map.get(experiment.id)};"
                  class="mx-auto"
                />
              </th>
              {#each hyperparams as key}
                <td class="p-3 text-ctp-text whitespace-nowrap">
                  {idToHP.get(experiment.id).get(key) ?? "-"}
                </td>
              {/each}
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </div>

  <div class="mt-8">
    <div
      class="text-sm font-semibold text-ctp-text p-4 border-b border-ctp-surface1"
    >
      <div class="flex items-center gap-2">Metrics Comparison</div>
    </div>

    <!-- Metric Selector -->
    <div class="p-4 border-b border-ctp-surface1">
      <!-- Dropdown selector -->
      <details class="relative">
        <summary
          class="flex items-center justify-between cursor-pointer p-2 hover:bg-ctp-surface1 transition-colors"
        >
          <span class="text-sm text-ctp-text">
            Select metrics ({selectedMetrics.length} of {commonMetrics.length})
          </span>
          <ChevronDown size={16} class="text-ctp-subtext1" />
        </summary>

        <div
          class="absolute top-full left-0 right-0 mt-1 z-10 max-h-60 overflow-y-auto border border-ctp-surface1 bg-ctp-surface0 rounded shadow-lg"
        >
          <!-- Search filter -->
          <div class="p-2 border-b border-ctp-surface1">
            <input
              type="search"
              placeholder="Filter metrics..."
              bind:value={searchFilter}
              class="w-full px-2 py-1 text-sm bg-ctp-base border border-ctp-surface1 rounded text-ctp-text placeholder-ctp-subtext0 focus:outline-none focus:border-ctp-blue"
            />
          </div>

          <!-- Control buttons -->
          <div class="flex gap-2 p-2 border-b border-ctp-surface1">
            <button
              onclick={selectAllMetrics}
              class="px-2 py-1 text-xs bg-ctp-green/20 text-ctp-green rounded hover:bg-ctp-green/30 transition-colors"
            >
              Select All
            </button>
            <button
              onclick={clearAllMetrics}
              class="px-2 py-1 text-xs bg-ctp-red/20 text-ctp-red rounded hover:bg-ctp-red/30 transition-colors"
            >
              Clear All
            </button>
          </div>

          <!-- Metric checkboxes -->
          <div class="p-1">
            {#each filteredMetrics as metric}
              <label
                class="flex items-center gap-2 p-2 hover:bg-ctp-surface1 rounded cursor-pointer"
              >
                <input
                  type="checkbox"
                  checked={selectedMetrics.includes(metric)}
                  onchange={() => toggleMetric(metric)}
                  class="text-ctp-blue focus:ring-ctp-blue focus:ring-2"
                />
                <span class="text-sm text-ctp-text">{metric}</span>
              </label>
            {/each}

            {#if filteredMetrics.length === 0}
              <div class="p-2 text-sm text-ctp-subtext0 text-center">
                No metrics found
              </div>
            {/if}
          </div>
        </div>
      </details>
    </div>
    <div class="w-full p-6">
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
            <Circle size={48} class="mx-auto mb-4 opacity-50" />
            <p class="text-lg mb-2">No metrics selected</p>
            <p class="text-sm">
              Select metrics above to start comparing experiments
            </p>
          </div>
        </div>
      {/if}
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
